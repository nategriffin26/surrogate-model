// The open cases, which one is shown, and the view they are all seen through. Each case has one status with its
// moves in one place:
//   example:  loading → ready | failed
//   file:     queued → analysing → ready, or failed from queued or analysing
//   variant:  ready, predicted before it opens (variants.js)
// and any case can be closed, which cancels its analysis. A file goes to the engine as soon as it is opened; the
// engine analyses one at a time in that order, and a file stays queued until its first stage begins.

import { loadResult } from '../engine/result.js';
import { Surface } from '../viewer/surface.js';
import { EXAMPLE_BASE } from './examples.js';

const MOVES = {
	queued: ['analysing', 'failed'],
	loading: ['ready', 'failed'],
	analysing: ['ready', 'failed'],
	ready: [],
	failed: [],
};

export const STAGES = ['reading', 'measuring', 'meshing', 'predicting'];

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
			/** The section: whether it is drawn and plotted, the axis its plane is normal to, and where it crosses it, in metres. */
			section: { on: true, axis: 1, offset: 0 },
			dock: 'section',
			dockCollapsed: false,
			/** The probe set in use: the one clicks add to and the probe table shows; or null. */
			probeSet: null,
			/** What the car on screen is compared with, {with: 'cfd' or a case id, sideBySide}; or null. */
			compare: null,
		};
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
		if (!c)
			throw new Error(
				`There is no open case ${id}. Open cases: ${this.cases.map((x) => `${x.id} (${x.name})`).join(', ') || 'none'}.`,
			);
		return c;
	}

	/** The case with an id, else the one on screen; either must have a result, or the error says why it has none. */
	withResult(id) {
		const c = id === undefined ? this.active : this.byId(id);
		if (c?.result) return c;
		throw new Error(
			id === undefined
				? 'No case with a result is on screen. Open a file or an example first.'
				: `Case ${id} has no result yet (${c.status}).`,
		);
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

	/** Open a case, after the case at `after` when given, and show it. */
	add(c, after = null) {
		const at = after ? this.cases.indexOf(after) + 1 : this.cases.length;
		this.cases.splice(at, 0, c);
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
			/** Each probe's site on this car, by probe, cached by the probes. */
			sites: null,
			/** Whether its field has made its entrance. */
			arrived: false,
			/** Section cuts through its working mesh, by plane, cached by the section tool. */
			cuts: null,
			/** For a variant, the car it was made from and the changes from that car: {root, changes}; else null. */
			variant: null,
			/** A car with variants: its surface predicted unchanged the way they are (a quick Result), so they compare like for like. */
			baseline: null,
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
			const result = await loadResult(new URL(id, EXAMPLE_BASE).href);
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

	/**
	 * Open a variant, a quick Result of a changed copy of `root`'s surface, after `root` and the variants of it already
	 * open. Its surface is kept in the workspace at `path`.
	 */
	openVariant(result, { name, root, changes, path }) {
		const c = { ...this.newCase(name, 'variant', null, 'ready', path), variant: { root, changes } };
		c.result = result;
		c.surface = new Surface(result.mesh);
		const last = this.cases.findLast((x) => x === root || x.variant?.root === root);
		return this.add(c, last);
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
				this.notify(c, 'full');
			},
			fullFields: (fields, fraction) => {
				c.surface.setFullFields(fields, fraction);
				this.notify(c, 'full');
			},
			// The forces, accuracy and totals summed over every polygon of the file, replacing the sampled ones.
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

	/**
	 * Analyse a ready file case's file again, so the engine holds its polygons once more: it holds the car analysed
	 * last only, and taking another file's CFD needs them. The car stays on screen while it runs; the new result, the
	 * same numbers summed over every polygon, then replaces the old, with the working mesh drawn from it and the
	 * cuts and probe sites made on the old one dropped. Resolves once every polygon is predicted.
	 */
	async analyseAgain(c, file) {
		let everyPolygon;
		const exact = new Promise((resolve) => (everyPolygon = resolve));
		const result = await this.engine.analyze(file, { exact: everyPolygon }, { signal: c.abort.signal });
		c.exact = await exact;
		c.result = { ...result, ...c.exact };
		c.surface.setWorking(result.mesh);
		c.cuts = null;
		c.sites = null;
		this.notify(c, 'geometry');
		this.notify(c, 'numbers');
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
