// Surrogate Model on the public site: a Cloudflare Worker on the routes nategriffin.dev/surrogate and
// nategriffin.dev/surrogate/*, which run before the site's own Worker. Its static assets are web/, staged under
// surrogate/ by deploy.sh so their paths are the public ones; wrangler.jsonc sends only the tool's page, the
// assistant's route and its kernel script through this module, and serves every other file straight from the assets.
// It records each page view and each answered assistant request in the site's visit log (D1 table visits, schema in
// the site's repository): time, path, referrer, city, region, country, the network's owner and user agent, never the
// IP address. The write runs after the response, and a failed write is only logged.
import { AssistantLimits, route } from './assistant.js';

export { AssistantLimits };

const HOST = 'nategriffin.dev';
const ASSISTANT = '/surrogate/api/assistant';

export default {
	async fetch(request, env, ctx) {
		const response = (await route(request, env)) ?? (await env.ASSETS.fetch(request));
		if (isVisit(request, response)) {
			ctx.waitUntil(record(request, env.DB).catch((error) => console.error('Visit not recorded:', error)));
		}
		return response;
	},
};

// On the live domain: the tool's page opened as the top-level document, served or revalidated, and not a prefetch;
// or a message the assistant answered. The page's own check that the assistant is available (GET) is not a visit.
function isVisit(request, response) {
	const url = new URL(request.url);
	if (url.hostname !== HOST) return false;
	if (url.pathname === ASSISTANT) return request.method === 'POST' && response.status === 200;
	const dest = request.headers.get('sec-fetch-dest');
	const purpose = request.headers.get('sec-purpose') ?? request.headers.get('purpose') ?? '';
	return (
		request.method === 'GET' &&
		url.pathname.endsWith('/') &&
		(response.status === 200 || response.status === 304) &&
		(dest === null || dest === 'document') &&
		!purpose.includes('prefetch')
	);
}

async function record(request, db) {
	const url = new URL(request.url);
	const cf = request.cf ?? {};
	await db
		.prepare('INSERT INTO visits (at, path, referrer, country, region, city, org, ua) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
		.bind(
			new Date().toISOString(),
			url.pathname + url.search,
			request.headers.get('referer'),
			cf.country ?? null,
			cf.region ?? null,
			cf.city ?? null,
			cf.asOrganization ?? null,
			request.headers.get('user-agent'),
		)
		.run();
}
