// The assistant's route alone, as a Worker of its own: `wrangler dev` in this folder runs it on this machine.
import { AssistantLimits, route } from './assistant.js';

export { AssistantLimits };

export default {
	async fetch(request, env) {
		return (await route(request, env)) ?? new Response('Not found', { status: 404 });
	},
};
