// The assistant's loop, run in the page: send the conversation and the tools to the model through the proxy
// (`/api/assistant`, src/surrogate/server.py), stream the reply, run the tools it calls, send their results, and
// repeat until it answers without calling one.

const ENDPOINT = new URL('../api/assistant', import.meta.url);

/** Whether this copy of the tool serves the assistant: 'available', 'unavailable' (no key, switched off, or a host without the route) or 'unreachable'. */
export async function availability() {
	try {
		const response = await fetch(ENDPOINT, { cache: 'no-store' });
		const body = response.headers.get('Content-Type')?.includes('json') ? await response.json() : null;
		return response.ok && body?.enabled ? 'available' : 'unavailable';
	} catch {
		return 'unreachable';
	}
}

/**
 * Run the assistant until it answers. Its messages and the tools' results are appended to `messages` as they come,
 * so the conversation stays one the model accepts even when the engineer stops it part way.
 * @param {object[]} messages  The conversation in the chat-completions format, system prompt first.
 * @param {{ tools: () => object[], execute: (call: object) => Promise<string>, signal: AbortSignal,
 *   onRequest: () => void, onReasoning: (text: string) => void, onText: (text: string) => void,
 *   onToolCall: (call: object) => void, onRequestEnd: () => void }} handlers
 *   `tools` is asked before every request, so operations added meanwhile are offered; `execute` runs one tool call
 *   (the same object `onToolCall` announced as soon as its name arrived) and returns what the model is told. The
 *   other callbacks follow one request as it streams: sent, a piece of its thinking ('' when the thinking arrives
 *   without text), a piece of its reply, a tool call begun, and over (whole, stopped or failed).
 */
export async function respond(messages, { tools, execute, signal, onRequest, onReasoning, onText, onToolCall, onRequestEnd }) {
	for (;;) {
		const message = { role: 'assistant', content: '', tool_calls: [], reasoning_details: [] };
		let whole = false;
		try {
			onRequest();
			await stream(messages, tools(), message, { onReasoning, onText, onToolCall, signal });
			whole = true;
		} finally {
			onRequestEnd();
			// A reply cut short (stopped, refused, broken off) keeps the text the engineer saw, and nothing else.
			if (whole) messages.push(sendable(message));
			else if (message.content) messages.push({ role: 'assistant', content: message.content });
		}
		if (message.tool_calls.length === 0) return;
		for (const call of message.tool_calls) {
			const content = signal.aborted ? JSON.stringify({ error: 'The engineer stopped the run before this call ran.' }) : await execute(call);
			messages.push({ role: 'tool', tool_call_id: call.id, content });
		}
		signal.throwIfAborted();
	}
}

/** The streamed message as the model is sent it next time. */
function sendable({ content, tool_calls, reasoning_details }) {
	return {
		role: 'assistant',
		content: content || (tool_calls.length ? null : ''),
		...(tool_calls.length && { tool_calls }),
		...(reasoning_details.length && { reasoning_details }),
	};
}

/** One request: read the server-sent events into `message` as they arrive. */
async function stream(messages, tools, message, { onReasoning, onText, onToolCall, signal }) {
	const response = await reach(() => fetch(ENDPOINT, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ messages, tools }),
		signal,
	}), "The tool's server did not answer, so the message did not reach the model.");
	if (!response.ok) throw new Error(await refusal(response));
	const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
	let buffer = '';
	let finish = null;
	for (;;) {
		const { value, done } = await reach(() => reader.read(), 'The connection to the model broke off during the reply.');
		if (done) break;
		buffer += value;
		for (let end = buffer.indexOf('\n'); end >= 0; end = buffer.indexOf('\n')) {
			const line = buffer.slice(0, end).trim();
			buffer = buffer.slice(end + 1);
			// Blank lines end events; lines starting ':' are the service's keep-alive comments.
			if (!line.startsWith('data:')) continue;
			const data = line.slice(5).trim();
			if (data === '[DONE]') continue;
			const chunk = JSON.parse(data);
			if (chunk.error) throw new Error(`The model stopped with an error: ${chunk.error.message}`);
			const choice = chunk.choices?.[0];
			if (!choice) continue;
			finish = choice.finish_reason ?? finish;
			const { content, tool_calls, reasoning_details } = choice.delta ?? {};
			// Thinking arrives as text (Claude's summary of it) or encrypted, which the engineer cannot read.
			for (const detail of reasoning_details ?? []) {
				addReasoning(message.reasoning_details, detail);
				onReasoning(detail.text ?? detail.summary ?? '');
			}
			if (content) {
				message.content += content;
				onText(content);
			}
			for (const part of tool_calls ?? []) {
				const call = (message.tool_calls[part.index ?? 0] ??= { id: '', type: 'function', function: { name: '', arguments: '' } });
				const named = !call.function.name && part.function?.name;
				if (part.id) call.id = part.id;
				if (part.function?.name) call.function.name += part.function.name;
				if (part.function?.arguments) call.function.arguments += part.function.arguments;
				if (named) onToolCall(call);
			}
		}
	}
	if (finish === 'length') {
		throw new Error('The reply reached its length limit and was cut off. Ask for less at a time.');
	}
}

/**
 * Reasoning arrives in pieces; pieces with the same index are one block, sent back whole with the message so the
 * model continues its own reasoning after a tool call.
 */
function addReasoning(details, piece) {
	const block = piece.index !== undefined && details.find((d) => d.index === piece.index && d.type === piece.type);
	if (!block) {
		details.push({ ...piece });
		return;
	}
	for (const key of ['text', 'summary', 'data']) if (piece[key]) block[key] = (block[key] ?? '') + piece[key];
	if (piece.signature) block.signature = piece.signature;
}

/** The message of a refusal: the proxy's or the model service's `{ error: { message } }`, else the status. */
async function refusal(response) {
	const body = await response.json().catch(() => null);
	return body?.error?.message ?? `The assistant's service answered ${response.status} ${response.statusText}.`;
}

/** `act`, with a network failure (the browser's bare "Failed to fetch") said as `message`; a stop stays a stop. */
async function reach(act, message) {
	try {
		return await act();
	} catch (error) {
		if (error.name === 'AbortError') throw error;
		throw new Error(message);
	}
}
