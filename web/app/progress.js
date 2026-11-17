// Progress of an analysis: the engine's four stages as they run, what runs the networks, and a way to cancel.
// A card in the middle of the view until the car appears, then a strip along the bottom so the car can be handled.

import { h } from './dom.js';
import { bytes, seconds } from './format.js';
import { icon } from './icons.js';
import { STAGES } from './session.js';

const LABELS = { reading: 'Reading', measuring: 'Measuring', meshing: 'Meshing', predicting: 'Predicting' };

export function createProgress({ cancel }) {
	const name = h('span', { class: 'progress-file' });
	const size = h('span', { class: 'progress-size num' });
	const device = h('span', { class: 'progress-device' });
	const rows = STAGES.map((stage) => {
		const fill = h('span', { class: 'progress-fill' });
		const time = h('span', { class: 'progress-time num' });
		const row = h('div', { class: 'progress-row', 'data-stage': stage },
			h('span', { class: 'progress-label' }, LABELS[stage]), time, h('span', { class: 'progress-bar' }, fill));
		return { stage, row, fill, time };
	});
	const stop = h('button', { type: 'button', class: 'button progress-cancel' }, icon('cancel'), 'Cancel');
	const el = h('div', { class: 'progress', role: 'status' },
		h('div', { class: 'progress-head' }, name, size),
		h('div', { class: 'progress-stages' }, rows.map((r) => r.row)),
		h('div', { class: 'progress-foot' }, device, stop));
	let current = null;
	stop.addEventListener('click', () => current && cancel(current));

	function render(c, runsOn) {
		current = c;
		name.textContent = c.name;
		size.textContent = c.bytes === null ? '' : bytes(c.bytes);
		device.textContent = runsOn ? `Networks on ${runsOn}` : '';
		device.classList.toggle('cpu', Boolean(runsOn?.startsWith('CPU')));
		const now = performance.now();
		for (const r of rows) {
			const st = c.stages[r.stage];
			const state = !st ? 'waiting' : st.end ? 'done' : 'active';
			r.row.dataset.state = c.status === 'queued' ? 'waiting' : state;
			r.fill.style.transform = `scaleX(${st ? (st.end ? 1 : st.fraction) : 0})`;
			r.time.textContent = st ? seconds(((st.end ?? now) - st.start) / 1000) : c.status === 'queued' && r.stage === 'reading' ? 'Queued' : '';
		}
	}

	/** Overall share done, for the band that crosses the lattice. */
	function fraction(c) {
		const done = STAGES.filter((stage) => c.stages[stage]?.end).length;
		const active = c.stage && !c.stages[c.stage].end ? c.stages[c.stage].fraction : 0;
		return (done + active) / STAGES.length;
	}

	return { el, render, fraction };
}
