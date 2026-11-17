// The open cases and which one is shown. Each case has one status with its moves in one place:
//   example:  loading → ready | failed
//   file:     queued → analysing → ready | failed
// and any case can be closed, which cancels its analysis. The engine analyses one file at a time, so files wait
// their turn in order.

import { loadResult } from '../engine/result.js';
import { Surface } from '../viewer/surface.js';

const MOVES = {
	queued: ['analysing'],
	loading: ['ready', 'failed'],
	analysing: ['ready', 'failed'],
	ready: [],
	failed: [],
};

export const STAGES = ['reading', 'measuring', 'meshing', 'predicting'];
const EXAMPLES = new URL('../examples/', import.meta.url);

export class Session extends EventTarget {
	constructor(engine) {
		super();
		this.engine = engine;
		this.cases = [];
		this.active = null;
		this.queue = Promise.resolve();
		this.serial = 0;
		this.device = null;
		engine.device.then((device) => {
			this.device = device;
			this.notify(null, 'device');
		});
	}

	/** detail.what: 'cases' (opened, closed), 'active', 'status', 'stage', 'geometry', 'full', 'numbers', 'device'. */
	notify(c, what) {
		this.dispatchEvent(new CustomEvent('change', { detail: { case: c, what } }));
	}

	move(c, status) {
		if (!MOVES[c.status].includes(status)) throw new Error(`A case cannot go from ${c.status} to ${status}`);
		c.status = status;
		this.notify(c, 'status');
	}

	add(c) {
		this.cases.push(c);
		this.notify(c, 'cases');
		this.activate(c);
		return c;
	}

	newCase(name, kind, bytes, status) {
		return {
			id: ++this.serial,
			name,
			kind,
			bytes,
			status,
			stage: null,
			stages: {},
			error: null,
			result: null,
			surface: null,
			full: null,
			view: null,
			abort: new AbortController(),
		};
	}

	activate(c) {
		if (c === this.active) return;
		this.active = c;
		this.notify(c, 'active');
	}

	async openExample(id, label) {
		const c = this.add(this.newCase(label, 'example', null, 'loading'));
		try {
			const result = await loadResult(new URL(id, EXAMPLES).href);
			if (c.closed) return;
			c.result = result;
			c.surface = new Surface(result.mesh);
			this.move(c, 'ready');
		} catch (error) {
			if (c.closed) return;
			c.error = `${label} could not be loaded (${error.message}). Check the connection and open it again.`;
			this.move(c, 'failed');
		}
	}

	openFile(file) {
		const c = this.add(this.newCase(file.name, 'file', file.size, 'queued'));
		this.queue = this.queue.then(() => (c.closed ? null : this.analyse(c, file)));
		return c;
	}

	async analyse(c, file) {
		this.move(c, 'analysing');
		const on = {
			stage: (stage, fraction) => {
				const now = performance.now();
				if (c.stage !== stage) {
					if (c.stage) c.stages[c.stage].end = now;
					c.stage = stage;
					c.stages[stage] = { start: now, end: null, fraction };
				}
				c.stages[stage].fraction = fraction;
				this.notify(c, 'stage');
			},
			geometry: (mesh) => {
				if (c.surface) c.surface.setWorking(mesh);
				else c.surface = new Surface(mesh);
				this.notify(c, 'geometry');
			},
			fullMesh: (mesh) => {
				c.surface.setFull(mesh);
				c.full = { fraction: 0 };
				this.notify(c, 'full');
			},
			fullFields: (fields, fraction) => {
				c.surface.setFullFields(fields, fraction);
				c.full.fraction = fraction;
				this.notify(c, 'full');
			},
			// Drag, agreement and coverage summed over every polygon of the file, replacing the sampled ones.
			exact: (numbers) => {
				c.exact = numbers;
				if (c.result) Object.assign(c.result, numbers);
				this.notify(c, 'numbers');
			},
		};
		try {
			const result = await this.engine.analyze(file, on, { signal: c.abort.signal });
			if (c.stage) c.stages[c.stage].end = performance.now();
			c.result = c.exact ? { ...result, ...c.exact } : result;
			if (c.surface) c.surface.setWorking(result.mesh);
			else c.surface = new Surface(result.mesh);
			this.move(c, 'ready');
		} catch (error) {
			if (c.closed) return;
			c.error = error.message;
			this.move(c, 'failed');
		}
	}

	/** Close a case: its analysis stops, its GPU memory is released, and a neighbouring case is shown. */
	close(c) {
		const i = this.cases.indexOf(c);
		if (i < 0) return;
		c.closed = true;
		c.abort.abort();
		c.surface?.dispose();
		this.cases.splice(i, 1);
		this.notify(c, 'cases');
		if (this.active === c) this.activate(this.cases[Math.min(i, this.cases.length - 1)] ?? null);
	}
}
