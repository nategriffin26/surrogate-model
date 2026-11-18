// The open cases, which one is shown, and the view they are all seen through. Each case has one status with its
// moves in one place:
//   example:  loading → ready | failed
//   file:     queued → analysing → ready, or failed from queued or analysing
// and any case can be closed, which cancels its analysis. A file goes to the engine as soon as it is opened; the
// engine analyses one at a time in that order, and a file stays queued until its first stage begins.

import { loadResult } from '../engine/result.js';
import { Surface } from '../viewer/surface.js';

const MOVES = {
	queued: ['analysing', 'failed'],
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
		this.serial = 0;
		this.device = null;
		/**
		 * The one view every case is seen through, so flipping tabs compares cars from the same angle on the same
		 * colour scale. The colour maps start from the first car with a result and then stay put.
		 */
		this.view = {
			field: 'cp',
			maps: null,
			display: { edges: false, ground: true },
			/** The live section; the sections kept in the plot beside it; the plot against metres or a fraction of the car; the car shown or all. */
			section: { on: true, axis: 1, offset: 0, clip: 0, kept: [], abscissa: 'metres', cars: 'shown' },
			dock: 'section',
			dockCollapsed: false,
			/** The faces picked out, as rules ({terms}: see regions.js), so every car shown picks the same part; or null. */
			selection: null,
			/** The probe set in use: the one clicks add to and the probe table shows; or null. */
			probeSet: null,
		};
		/** Saved regions, {id, name, terms}, which any car can be measured on. */
		this.regions = [];
		/** Probe sets, {id, name, probes: [{n, point, label?}], next}, which every car reads at its own cells. */
		this.probeSets = [];
		engine.device.then((device) => {
			this.device = device;
			this.notify(null, 'device');
		});
	}

	/** The case with an id, or an error naming the open ones. */
	byId(id) {
		const c = this.cases.find((x) => x.id === id);
		if (!c) throw new Error(`There is no open case ${id}. Open cases: ${this.cases.map((x) => `${x.id} (${x.name})`).join(', ') || 'none'}.`);
		return c;
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

	/** A file case keeps the workspace path of the file it shows. */
	newCase(name, kind, bytes, status, path = null) {
		return {
			id: ++this.serial,
			name,
			kind,
			path,
			bytes,
			status,
			stage: null,
			stages: {},
			error: null,
			result: null,
			surface: null,
			full: null,
			/** Each probe's site on this car, by probe, cached by the probes. */
			sites: null,
			/** Whether its field has made its entrance. */
			arrived: false,
			/** Section cuts through its working mesh, by plane, cached by the section tool. */
			cuts: null,
			abort: new AbortController(),
		};
	}

	activate(c) {
		if (c === this.active) return;
		this.active = c;
		this.notify(c, 'active');
	}

	async openExample(id, label) {
		const c = this.add({ ...this.newCase(label, 'example', null, 'loading'), example: id });
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

	/** Analyse a geometry from the workspace: `file` is its blob, as a File named for it. */
	openFile(file, path) {
		const c = this.add(this.newCase(file.name, 'file', file.size, 'queued', path));
		this.analyse(c, file);
		return c;
	}

	async analyse(c, file) {
		const on = {
			stage: (stage, fraction) => {
				if (c.status === 'queued') this.move(c, 'analysing');
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
