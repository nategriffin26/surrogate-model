// The public assistant route (deploy/assistant-worker/assistant.js): its limits, refusals and pass-through, with the
// model service stubbed by a stream recorded from OpenRouter. Run with: deno test --allow-read tests/assistant_worker.test.js
import { deepStrictEqual as assertEquals, ok as assert } from 'node:assert/strict';
import { AssistantLimits, KERNEL_POLICY, LIMITS, Limiter, MAX_REQUEST_BYTES, STREAM_TYPE, TOOL_PATH, route } from '../deploy/assistant-worker/assistant.js';

const RECORDED = await Deno.readFile(new URL('./fixtures/assistant_tool_call.sse', import.meta.url));
const ORIGIN = 'https://nategriffin.dev';
const API = `https://nategriffin.dev${TOOL_PATH}api/assistant`;
const CONVERSATION = { messages: [{ role: 'user', content: 'What is the lowest Cp on case 1?' }], tools: [] };
const NOON = Date.parse('2026-11-19T12:00:00Z');

/** Durable Object storage in a Map, shared by every instance made from it, as a real object's storage is. */
function objectContext(storage = new Map()) {
	return {
		storage: { get: async (key) => structuredClone(storage.get(key)), put: async (key, value) => void storage.set(key, structuredClone(value)) },
		blockConcurrencyWhile: (work) => work(),
	};
}

function fakeEnv(overrides = {}) {
	const limits = new AssistantLimits(objectContext());
	return {
		OPENROUTER_API_KEY: 'sk-test-key',
		ASSISTANT_MODEL: 'meta/muse-spark-1.3-contributor',
		ASSISTANT_LIMITS: { idFromName: (name) => name, get: () => ({ fetch: (url, init) => limits.fetch(new Request(url, init)) }) },
		ASSETS: { fetch: async () => new Response('// kernel', { headers: { 'Content-Type': 'text/javascript' } }) },
		...overrides,
	};
}

/** Stub the model service for one test: records what it was sent and answers with `reply()`. */
function stubModelService(reply = () => new Response(RECORDED, { headers: { 'Content-Type': 'text/event-stream' } })) {
	const sent = [];
	const real = globalThis.fetch;
	globalThis.fetch = async (url, init) => {
		sent.push({ url, headers: init.headers, body: JSON.parse(init.body) });
		return reply();
	};
	return { sent, restore: () => (globalThis.fetch = real) };
}

function post(body, { origin = ORIGIN, visitor = '203.0.113.7' } = {}) {
	return new Request(API, { method: 'POST', headers: { Origin: origin, 'CF-Connecting-IP': visitor, 'Content-Type': 'application/json' }, body });
}

Deno.test('a visitor over 30 requests a minute is refused until the next minute, others are not', () => {
	const limiter = new Limiter();
	for (let i = 0; i < LIMITS.perVisitorPerMinute; i++) assertEquals(limiter.admit('a', NOON + i).refused, null);
	assertEquals(limiter.admit('a', NOON + 1000), { refused: 'visitor', retryAfter: 59 });
	assertEquals(limiter.admit('b', NOON + 1000).refused, null);
	assertEquals(limiter.admit('a', NOON + 60000).refused, null);
});

Deno.test('the 121st request in a minute is refused whoever sends it', () => {
	const limiter = new Limiter();
	for (let i = 0; i < LIMITS.perMinute; i++) assertEquals(limiter.admit(`visitor ${i % 5}`, NOON).refused, null);
	assertEquals(limiter.admit('someone new', NOON).refused, 'minute');
	assertEquals(limiter.admit('someone new', NOON + 60000).refused, null);
});

Deno.test('the 5001st request in a UTC day is refused until midnight', () => {
	const limiter = new Limiter();
	for (let i = 0; i < LIMITS.perDay; i++) assertEquals(limiter.admit('a', NOON + Math.floor(i / 20) * 60000).refused, null);
	const late = Date.parse('2026-11-19T23:59:00Z');
	assertEquals(limiter.admit('a', late), { refused: 'day', retryAfter: 60 });
	assertEquals(limiter.admit('a', Date.parse('2026-11-20T00:00:01Z')).refused, null);
});

Deno.test("the day's count survives the limits object being evicted", async () => {
	const storage = new Map();
	const first = new AssistantLimits(objectContext(storage));
	const admit = (object) => object.fetch(new Request('https://limits/admit', { method: 'POST', body: JSON.stringify({ visitor: 'a' }) }));
	for (let i = 0; i < 3; i++) await admit(first);
	const second = new AssistantLimits(objectContext(storage));
	await admit(second);
	assertEquals(second.limiter.inDay, 4);
});

Deno.test('over the limit the route answers 429 with a message and Retry-After, and the model is not called', async () => {
	const realNow = Date.now;
	Date.now = () => NOON;
	const service = stubModelService();
	try {
		const env = fakeEnv();
		for (let i = 0; i < LIMITS.perVisitorPerMinute; i++) assertEquals((await route(post(JSON.stringify(CONVERSATION)), env)).status, 200);
		const refused = await route(post(JSON.stringify(CONVERSATION)), env);
		assertEquals(refused.status, 429);
		assertEquals(refused.headers.get('Retry-After'), '60');
		assert((await refused.json()).error.message.startsWith('Too many requests'));
		assertEquals(service.sent.length, LIMITS.perVisitorPerMinute);
	} finally {
		service.restore();
		Date.now = realNow;
	}
});

Deno.test('the stream passes through unchanged, with the model and key added by the route', async () => {
	const service = stubModelService();
	try {
		const response = await route(post(JSON.stringify({ ...CONVERSATION, model: 'openai/some-expensive-model', max_tokens: 1e6 })), fakeEnv());
		assertEquals(response.status, 200);
		assertEquals(response.headers.get('Content-Type'), STREAM_TYPE);
		assertEquals(new Uint8Array(await response.arrayBuffer()), RECORDED);
		const [{ headers, body }] = service.sent;
		assertEquals(headers.Authorization, 'Bearer sk-test-key');
		assertEquals(body, { ...CONVERSATION, model: 'meta/muse-spark-1.3-contributor', stream: true, reasoning: { effort: 'high' }, max_tokens: 16000, cache_control: { type: 'ephemeral' } });
	} finally {
		service.restore();
	}
});

Deno.test("a refusal by the model service reaches the page as its own status and body", async () => {
	const refusal = { error: { code: 402, message: 'This key has used its credit limit.' } };
	const service = stubModelService(() => Response.json(refusal, { status: 402 }));
	try {
		const response = await route(post(JSON.stringify(CONVERSATION)), fakeEnv());
		assertEquals(response.status, 402);
		assertEquals(await response.json(), refusal);
	} finally {
		service.restore();
	}
});

Deno.test('without a key or switched off the route answers 503, and other sites and oversized bodies are refused', async () => {
	const service = stubModelService();
	try {
		for (const env of [fakeEnv({ OPENROUTER_API_KEY: undefined }), fakeEnv({ ASSISTANT_ENABLED: '0' })]) {
			assertEquals((await route(new Request(API), env)).status, 503);
			const refused = await route(post(JSON.stringify(CONVERSATION)), env);
			assertEquals(refused.status, 503);
			assertEquals((await refused.json()).error.message, 'The assistant is not available in this copy.');
		}
		assertEquals((await route(post(JSON.stringify(CONVERSATION), { origin: 'https://elsewhere.example' }), fakeEnv())).status, 403);
		assertEquals((await route(post(JSON.stringify({ ...CONVERSATION, padding: 'x'.repeat(MAX_REQUEST_BYTES) })), fakeEnv())).status, 413);
		assertEquals((await route(post(JSON.stringify({ messages: [] })), fakeEnv())).status, 400);
		assertEquals(service.sent.length, 0);
		assertEquals(await (await route(new Request(API), fakeEnv())).json(), { enabled: true, model: 'meta/muse-spark-1.3-contributor' });
	} finally {
		service.restore();
	}
});

Deno.test("the kernel's worker script is served with its network policy and other paths are left to the site", async () => {
	const worker = await route(new Request(`https://nategriffin.dev${TOOL_PATH}assistant/kernel-worker.js`), fakeEnv());
	assertEquals(worker.headers.get('Content-Security-Policy'), KERNEL_POLICY);
	assertEquals(await worker.text(), '// kernel');
	assertEquals(await route(new Request(`https://nategriffin.dev${TOOL_PATH}index.html`), fakeEnv()), null);
});
