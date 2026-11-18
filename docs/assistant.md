# Assistant

The Assistant panel holds one agent the engineer talks to. Its tools are the operations the tool's own controls run (`web/app/operations.js`), so what it does shows on screen as if the engineer had done it, and a Python kernel.

A reply streams in as it is written:

- the assistant's thinking (Claude's summary of it) opens while it arrives and folds to how long it took;
- each tool call is a step that opens to show what was asked and what came back;
- whatever is in progress has its name shimmering.

The stop button ends a run at once and keeps what arrived.

## What it asks before doing

The assistant asks the engineer before it:

- closes a case;
- downloads an export or a file;
- deletes a file;
- removes a layer the engineer made;
- moves a file the engineer made;
- runs Python code that removes, replaces or moves the engineer's files.

Every other operation runs without asking.

## Python kernel

The kernel is Pyodide (Python 3.14 with numpy, scipy, pandas and matplotlib) in a worker of its own, loaded from the jsDelivr CDN on first use and kept for the conversation. Its working directory is the tool's workspace, the same files the Files panel shows. A run computing for more than 60 s is stopped by restarting the kernel.

It has no network: its script is served with a Content-Security-Policy that allows only the Pyodide CDN (`KERNEL_POLICY` in `src/surrogate/server.py`), and the request functions are removed once Pyodide has loaded.

Its module `pressure` gives the open cases as numpy arrays, runs any operation, predicts a changed geometry with the browser engine and writes `.vtp` files, so a study is a loop of deform, predict and measure. It draws on the car as a plotting library would: `case.show(values, name)` colours the car by an array, `case.paint(mask, 'orange')` paints faces, and `view.label`, `view.arrows`, `view.line` and the rest put marks on it, thousands in one call.

## Route to its model

The page sends the conversation to `/api/assistant`, and the server adds the key, the model, the reasoning effort and the output limit before forwarding it to OpenRouter and streaming the reply back.

- `OPENROUTER_API_KEY`, in the environment or in `local/secrets/assistant.env`, enables it.
- `ASSISTANT_MODEL` picks the model (default `anthropic/claude-opus-5.5`), and `ASSISTANT_ENABLED=0` switches it off.
- Without a key, or on a host without the route, the panel says the assistant is not available and the rest of the tool works.
- A host that serves the route but not the kernel's Content-Security-Policy does not start the Python kernel: `run_python` fails with a message saying so, and the rest of the tool works.

The tool is also published at https://nategriffin.dev/surrogate/ by `deploy/assistant-worker/deploy.sh`: a Cloudflare Worker (`worker.js`, `wrangler.jsonc`) serves `web/` as static files and the same route from `assistant.js`, with limits of 120 requests a minute overall, 60 a minute from one address and 5,000 a day, and a credit limit on its key at OpenRouter behind them.
