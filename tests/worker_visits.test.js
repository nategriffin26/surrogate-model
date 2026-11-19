// The public Worker's visit log (deploy/assistant-worker/worker.js): which requests it records, what a row holds, and
// that a failed write never changes the response. Run with: deno test --allow-read tests/worker_visits.test.js
import { deepStrictEqual as assertEquals } from 'node:assert/strict';
import worker from '../deploy/assistant-worker/worker.js';

const PAGE = 'https://nategriffin.dev/surrogate/';
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15';

/** A D1 binding that keeps the values of each insert, or fails every write with `error`. */
function fakeDb(error) {
	const rows = [];
	return {
		rows,
		prepare: () => ({
			bind: (...values) => ({
				run: async () => {
					if (error) throw error;
					rows.push(values);
				},
			}),
		}),
	};
}

/** Serves `request` the way the deployed Worker does and waits for the writes it handed to waitUntil. */
async function serve(request, { db = fakeDb(), assetStatus = 200, env = {} } = {}) {
	const pending = [];
	const assets = { fetch: async () => new Response('<!doctype html>', { status: assetStatus }) };
	const response = await worker.fetch(
		request,
		{ DB: db, ASSETS: assets, ...env },
		{ waitUntil: (p) => pending.push(p) },
	);
	await Promise.all(pending);
	return { response, rows: db?.rows };
}

function page(url = PAGE, headers = {}) {
	const request = new Request(url, {
		headers: { 'Sec-Fetch-Dest': 'document', 'User-Agent': UA, 'CF-Connecting-IP': '203.0.113.7', ...headers },
	});
	Object.defineProperty(request, 'cf', {
		value: { country: 'US', region: 'California', city: 'San Francisco', asOrganization: 'Rescale' },
	});
	return request;
}

Deno.test('opening the page records its path, place, network and browser, and not the address', async () => {
	const { response, rows } = await serve(page(`${PAGE}?case=1`, { Referer: 'https://nategriffin.dev/' }));
	assertEquals(response.status, 200);
	assertEquals(rows.length, 1);
	const [at, ...rest] = rows[0];
	assertEquals(typeof at, 'string');
	assertEquals(rest, [
		'/surrogate/?case=1',
		'https://nategriffin.dev/',
		'US',
		'California',
		'San Francisco',
		'Rescale',
		UA,
	]);
});

Deno.test('prefetches, frames, missing pages and other hosts are not visits', async () => {
	for (const request of [
		page(PAGE, { 'Sec-Purpose': 'prefetch' }),
		page(PAGE, { 'Sec-Fetch-Dest': 'iframe' }),
		page('https://surrogate-model.nategriffin26.workers.dev/surrogate/'),
	])
		assertEquals((await serve(request)).rows.length, 0);
	assertEquals((await serve(page(), { assetStatus: 404 })).rows.length, 0);
});

Deno.test("the page's check that the assistant is available is not a visit", async () => {
	const { response, rows } = await serve(page('https://nategriffin.dev/surrogate/api/assistant'));
	assertEquals(response.status, 503);
	assertEquals(rows.length, 0);
});

Deno.test('a message the assistant answers is a visit', async () => {
	const realFetch = globalThis.fetch;
	globalThis.fetch = async () => new Response('data: [DONE]\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
	try {
		const request = new Request('https://nategriffin.dev/surrogate/api/assistant', {
			method: 'POST',
			headers: { Origin: 'https://nategriffin.dev', 'User-Agent': UA },
			body: JSON.stringify({ messages: [{ role: 'user', content: 'Hi' }], tools: [] }),
		});
		const limits = { fetch: async () => Response.json({ refused: null }) };
		const env = { OPENROUTER_API_KEY: 'sk-test-key', ASSISTANT_LIMITS: { idFromName: () => 'all', get: () => limits } };
		const { response, rows } = await serve(request, { env });
		assertEquals(response.status, 200);
		assertEquals(
			rows.map((row) => row[1]),
			['/surrogate/api/assistant'],
		);
	} finally {
		globalThis.fetch = realFetch;
	}
});

Deno.test('a failed write, or no database, still serves the page unchanged', async () => {
	const realError = console.error;
	const logged = [];
	console.error = (...args) => logged.push(args);
	try {
		for (const db of [fakeDb(new Error('D1_ERROR: no such table: visits')), null]) {
			const { response } = await serve(page(), { db });
			assertEquals([response.status, await response.text()], [200, '<!doctype html>']);
		}
		assertEquals(logged.length, 2);
	} finally {
		console.error = realError;
	}
});
