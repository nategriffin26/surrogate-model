// Surrogate Model on the public site: a Cloudflare Worker on the routes nategriffin.dev/surrogate and
// nategriffin.dev/surrogate/*, which run before the site's own Worker. Its static assets are web/, staged under
// surrogate/ by deploy.sh so their paths are the public ones; wrangler.jsonc sends only the assistant's route and
// its kernel script through this module, and serves every other file straight from the assets.
import { AssistantLimits, route } from './assistant.js';

export { AssistantLimits };

export default {
	async fetch(request, env) {
		return (await route(request, env)) ?? env.ASSETS.fetch(request);
	},
};
