/**
 * The engine's helper workers (`prediction-worker.js`), one task at a time each, tasks handed to
 * whichever is free first; and how many workers the engine runs beside the page.
 */

/** Workers to run beside the page: one per core but the one the page keeps. */
export function helperCount() {
  return Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
}

/**
 * @param {number} size
 * @param {import('./network.js').Ensemble} ensemble
 */
export function createPool(size, ensemble) {
  const workers = Array.from({ length: size }, () => new Worker(new URL('./prediction-worker.js', import.meta.url), { type: 'module' }));
  const idle = [...workers];
  /** @type {{ id: number, message: object, transfer: Transferable[], resolve: Function, reject: Function }[]} */
  const queue = [];
  const running = new Map();
  let nextId = 1;

  for (const worker of workers) {
    worker.postMessage({ type: 'ensemble', ensemble });
    worker.onmessage = ({ data }) => {
      const task = running.get(data.id);
      running.delete(data.id);
      idle.push(worker);
      task.resolve(data.result);
      dispatch();
    };
    worker.onerror = (event) => {
      event.preventDefault();
      const error = new Error(`A helper worker failed: ${event.message}`);
      for (const task of running.values()) task.reject(error);
      running.clear();
    };
  }

  function dispatch() {
    while (idle.length && queue.length) {
      const worker = idle.pop();
      const task = queue.shift();
      running.set(task.id, task);
      worker.postMessage({ ...task.message, id: task.id }, task.transfer);
    }
  }

  return {
    size,
    /** Send every worker the same message, ahead of any task queued after it. */
    broadcast(message) {
      for (const worker of workers) worker.postMessage(message);
    },
    /** @returns {Promise<any>} the worker's result */
    run(message, transfer = []) {
      return new Promise((resolve, reject) => {
        queue.push({ id: nextId++, message, transfer, resolve, reject });
        dispatch();
      });
    },
    /** Drop the tasks not yet started; their promises reject with `reason`. */
    clear(reason) {
      for (const task of queue.splice(0)) task.reject(reason);
    },
  };
}
