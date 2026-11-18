/**
 * The assistant's model route on the public site, as a Cloudflare Worker: the contract of the local server's
 * `/api/assistant` (src/surrogate/server.py), plus what a public page needs: a request limit per minute overall
 * and per visitor, a limit per day, an origin check and a size cap. A credit limit set on the key at OpenRouter is
 * the hard stop behind them. It also serves the kernel's worker script with the policy that keeps the Python
 * kernel off the network.
 *
 * Mounting it in the site's Worker, with `web/` published under TOOL_PATH as the Worker's static assets (`ASSETS`):
 * - this file beside the site's main module, which imports `{ route, AssistantLimits }` from it, exports
 *   `AssistantLimits` (a Worker's main module exports only its handler and its Durable Object classes), and in
 *   `fetch` first runs `const handled = await route(request, env); if (handled) return handled;`
 * - in the site's wrangler configuration, the limits' Durable Object and its migration:
 *   `"durable_objects": { "bindings": [{ "name": "ASSISTANT_LIMITS", "class_name": "AssistantLimits" }] }` and
 *   `"migrations": [{ "tag": "assistant-limits", "new_sqlite_classes": ["AssistantLimits"] }]`
 * - the key as a secret: `wrangler secret put OPENROUTER_API_KEY`.
 * `vars.ASSISTANT_MODEL` overrides DEFAULT_MODEL. Switch the assistant off with `vars.ASSISTANT_ENABLED = "0"` (or
 * by deleting the secret) and deploy; the rest of the tool works without it.
 */

/** Requests a minute from everyone, a minute from one visitor (by IP address, held in memory only), and a day. */
export const LIMITS = { perMinute: 120, perVisitorPerMinute: 30, perDay: 5000 };
export const TOOL_PATH = '/surrogate-model/';
export const ALLOWED_ORIGINS = ['https://nategriffin.dev'];

// The rest of the contract, as in src/surrogate/server.py; tests/test_assistant_proxy.py holds the two equal.
export const UPSTREAM = 'https://openrouter.ai/api/v1/chat/completions';
export const DEFAULT_MODEL = 'anthropic/claude-opus-5.5';
export const REASONING_EFFORT = 'high';
export const MAX_OUTPUT_TOKENS = 16000;
export const MAX_REQUEST_BYTES = 512 * 1024;
export const UNAVAILABLE = 'The assistant is not available in this copy.';
export const STREAM_TYPE = 'text/event-stream; charset=utf-8';
export const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/';
export const KERNEL_POLICY = `default-src 'none'; script-src ${PYODIDE_CDN} 'wasm-unsafe-eval'; connect-src ${PYODIDE_CDN}`;

const REFUSALS = {
	visitor: 'Too many requests from this browser in the last minute. Try again in a minute.',
	minute: 'The assistant is busy. Try again in a minute.',
	day: "The assistant has reached today's limit. Try again tomorrow.",
};

/**
 * The response for the tool's assistant route and its kernel worker script; null for every other request, which
 * the site serves as before.
 * @param {Request} request
 * @param {{ ASSETS: { fetch(request: Request): Promise<Response> }, ASSISTANT_LIMITS: object, OPENROUTER_API_KEY?: string, ASSISTANT_MODEL?: string, ASSISTANT_ENABLED?: string }} env
 */
export async function route(request, env) {
	const path = new URL(request.url).pathname;
	if (path === `${TOOL_PATH}assistant/kernel-worker.js`) {
		const asset = await env.ASSETS.fetch(request);
		const response = new Response(asset.body, asset);
		response.headers.set('Content-Security-Policy', KERNEL_POLICY);
		return response;
	}
	if (path !== `${TOOL_PATH}api/assistant`) return null;

	const enabled = Boolean(env.OPENROUTER_API_KEY) && env.ASSISTANT_ENABLED !== '0';
	const model = env.ASSISTANT_MODEL || DEFAULT_MODEL;
	if (request.method === 'GET') {
		return enabled
			? Response.json({ enabled, model })
			: Response.json({ enabled, model: null, error: { message: UNAVAILABLE } }, { status: 503 });
	}
	if (request.method !== 'POST') return refuse(405, 'Use GET or POST.');
	if (!enabled) return refuse(503, UNAVAILABLE);
	if (!ALLOWED_ORIGINS.includes(request.headers.get('Origin')))
		return refuse(403, 'This route serves the tool on its own site only.');

	const declared = Number(request.headers.get('Content-Length') ?? 0);
	const bytes = declared > MAX_REQUEST_BYTES ? null : await request.arrayBuffer();
	if (bytes === null || bytes.byteLength > MAX_REQUEST_BYTES)
		return refuse(413, 'This conversation is too long to send. Start a new one.');
	let body;
	try {
		body = JSON.parse(new TextDecoder().decode(bytes));
	} catch {
		return refuse(400, 'The request body is not JSON.');
	}
	const { messages, tools } = body ?? {};
	if (!Array.isArray(messages) || messages.length === 0 || !Array.isArray(tools)) {
		return refuse(400, 'Send {messages, tools}: messages a non-empty list, tools a list.');
	}

	const limits = env.ASSISTANT_LIMITS.get(env.ASSISTANT_LIMITS.idFromName('all'));
	const visitor = request.headers.get('CF-Connecting-IP') ?? 'unknown';
	const verdict = await (
		await limits.fetch('https://limits/admit', { method: 'POST', body: JSON.stringify({ visitor }) })
	).json();
	if (verdict.refused) {
		return Response.json(
			{ error: { message: REFUSALS[verdict.refused] } },
			{ status: 429, headers: { 'Retry-After': String(verdict.retryAfter) } },
		);
	}

	let upstream;
	try {
		upstream = await fetch(UPSTREAM, {
			method: 'POST',
			headers: { Authorization: `Bearer ${env.OPENROUTER_API_KEY}`, 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model,
				messages,
				tools,
				stream: true,
				reasoning: { effort: REASONING_EFFORT },
				max_tokens: MAX_OUTPUT_TOKENS,
				cache_control: { type: 'ephemeral' },
			}),
		});
	} catch (error) {
		return refuse(502, `The model service could not be reached (${error.message}).`);
	}
	if (!upstream.ok) {
		return new Response(upstream.body, {
			status: upstream.status,
			headers: { 'Content-Type': upstream.headers.get('Content-Type') ?? 'application/json' },
		});
	}
	return new Response(upstream.body, { headers: { 'Content-Type': STREAM_TYPE, 'Cache-Control': 'no-cache' } });
}

function refuse(status, message) {
	return Response.json({ error: { message } }, { status });
}

/**
 * Counts in fixed windows: the current minute overall and per visitor, and the current day (UTC). A request is
 * counted only when admitted.
 */
export class Limiter {
	constructor(limits = LIMITS) {
		this.limits = limits;
		this.minute = -1;
		this.inMinute = 0;
		this.byVisitor = new Map();
		this.day = '';
		this.inDay = 0;
	}

	/** Admit a request at `now` (ms): `{ refused: null }`, or which limit refused it and the seconds until it resets. */
	admit(visitor, now) {
		const minute = Math.floor(now / 60000);
		const day = new Date(now).toISOString().slice(0, 10);
		if (minute !== this.minute) {
			this.minute = minute;
			this.inMinute = 0;
			this.byVisitor.clear();
		}
		if (day !== this.day) {
			this.day = day;
			this.inDay = 0;
		}
		const visitorCount = this.byVisitor.get(visitor) ?? 0;
		const toNextMinute = Math.ceil(((minute + 1) * 60000 - now) / 1000);
		if (this.inDay >= this.limits.perDay)
			return { refused: 'day', retryAfter: Math.ceil((Date.parse(`${day}T00:00:00Z`) + 86400000 - now) / 1000) };
		if (this.inMinute >= this.limits.perMinute) return { refused: 'minute', retryAfter: toNextMinute };
		if (visitorCount >= this.limits.perVisitorPerMinute) return { refused: 'visitor', retryAfter: toNextMinute };
		this.inMinute++;
		this.inDay++;
		this.byVisitor.set(visitor, visitorCount + 1);
		return { refused: null };
	}
}

/**
 * One instance holds every count, so the limits are global rather than per Cloudflare location. The day's count
 * is stored, so it survives the object being evicted; the minute's counts live in memory.
 */
export class AssistantLimits {
	constructor(ctx) {
		this.ctx = ctx;
		this.limiter = new Limiter();
		ctx.blockConcurrencyWhile(async () => {
			const saved = await ctx.storage.get('day');
			if (saved) Object.assign(this.limiter, saved);
		});
	}

	async fetch(request) {
		const { visitor } = await request.json();
		const verdict = this.limiter.admit(visitor, Date.now());
		if (!verdict.refused) await this.ctx.storage.put('day', { day: this.limiter.day, inDay: this.limiter.inDay });
		return Response.json(verdict);
	}
}
