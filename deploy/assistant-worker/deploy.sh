#!/bin/sh
# Publish Surrogate Model to https://nategriffin.dev/surrogate/: web/ as static files and the assistant's route
# (worker.js, wrangler.jsonc). Needs Bun and a wrangler login with access to the nategriffin.dev zone. Extra
# arguments go to wrangler deploy, for example --var ASSISTANT_MODEL:<model>.
#   deploy/assistant-worker/deploy.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
stage="$root/local/deploy/assets"
rm -rf "$stage"
mkdir -p "$stage"
cp -R "$root/web" "$stage/surrogate"
cd "$here"
exec bunx wrangler deploy "$@"
