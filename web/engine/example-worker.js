/**
 * Fetches and decodes one bundled example for `loadResult`, off the page's main thread, and hands
 * its arrays back by transfer.
 */

import { unpackResult } from './result.js';

self.onmessage = async ({ data: base }) => {
  try {
    const [summary, buffer] = await Promise.all([
      fetchOk(`${base}.json`).then((response) => response.json()),
      fetchOk(`${base}.bin`).then((response) => response.arrayBuffer()),
    ]);
    const result = await unpackResult(summary, buffer);
    self.postMessage({ result }, buffersOf(result));
  } catch (error) {
    self.postMessage({ message: error.message });
  }
};

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`The bundled example could not be loaded (${response.status} ${url}). Reload the page.`);
  return response;
}

/** Every distinct ArrayBuffer under `value`, for the transfer list. */
function buffersOf(value, found = new Set()) {
  if (ArrayBuffer.isView(value)) found.add(value.buffer);
  else if (value && typeof value === 'object') for (const item of Object.values(value)) buffersOf(item, found);
  return [...found];
}
