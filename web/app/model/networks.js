// The networks: one of them working on the point (its inputs as it reads them, its hidden layers, its Cp), and the
// cars each was trained on and held out from. Choosing a network in the grid shows its working.

import { h } from '../dom.js';
import { count, signed } from '../format.js';
import { bars, chips, countWord, section, shade } from './charts.js';
import { groupOf, groupsGiven, learnerName } from './groups.js';
import { locator } from './overview.js';

const HEIGHT = 250;
const TOP = 18;
const BOTTOM = 24;
const STRIP = 24;
const BAND = 5;
const NODE = 20;

/** The weights and biases of a network with these widths. */
const parameters = (layers) => layers.slice(1).reduce((sum, width, l) => sum + width * (layers[l] + 1), 0);

/** One network's values at the point as strips of units, inputs on the left beside their groups' colours, Cp on the right. */
function drawNetwork(canvas, { card, groups, layers, cp }) {
	const width = canvas.parentElement.clientWidth;
	if (!width) return;
	const ratio = devicePixelRatio;
	canvas.width = Math.round(width * ratio);
	canvas.height = Math.round(HEIGHT * ratio);
	canvas.style.height = `${HEIGHT}px`;
	const g = canvas.getContext('2d');
	g.setTransform(ratio, 0, 0, ratio, 0, 0);
	const span = HEIGHT - TOP - BOTTOM;
	const node = [width - NODE - 2, TOP + span / 2];
	const right = node[0] - NODE - 8;
	const gap = (right - BAND - layers.length * STRIP) / (layers.length - 1 + 0.6);
	const xs = layers.map((_, l) => BAND + 2 + l * (STRIP + gap));

	for (let l = 0; l < xs.length; l++) {
		const from = xs[l] + STRIP;
		const to = l + 1 < xs.length ? xs[l + 1] : node[0] - NODE;
		const fan = g.createLinearGradient(from, 0, to, 0);
		fan.addColorStop(0, 'rgba(11, 111, 191, 0.09)');
		fan.addColorStop(1, 'rgba(11, 111, 191, 0.04)');
		g.fillStyle = fan;
		g.beginPath();
		g.moveTo(from, TOP);
		g.lineTo(to, l + 1 < xs.length ? TOP : node[1] - NODE * 0.6);
		g.lineTo(to, l + 1 < xs.length ? TOP + span : node[1] + NODE * 0.6);
		g.lineTo(from, TOP + span);
		g.fill();
	}
	layers.forEach((values, l) => {
		const step = span / values.length;
		values.forEach((value, i) => {
			g.fillStyle = shade(value, l === 0 ? 1.5 : 1);
			g.fillRect(xs[l], TOP + i * step, STRIP, Math.max(step, 0.6) + 0.35);
		});
		g.strokeStyle = 'rgba(13, 34, 51, 0.18)';
		g.lineWidth = 1;
		g.strokeRect(xs[l] - 0.5, TOP - 0.5, STRIP + 1, span + 1);
	});
	const inputStep = span / layers[0].length;
	card.columns.forEach((_, i) => {
		g.fillStyle = groupOf(groups, i).color;
		g.fillRect(0, TOP + i * inputStep, BAND, inputStep + 0.35);
	});

	g.beginPath();
	g.arc(node[0], node[1], NODE, 0, 2 * Math.PI);
	g.fillStyle = '#fff';
	g.fill();
	g.lineWidth = 2;
	g.strokeStyle = '#0d2233';
	g.stroke();
	const font = getComputedStyle(canvas).fontFamily;
	g.fillStyle = '#0d2233';
	g.textAlign = 'center';
	g.textBaseline = 'middle';
	g.font = `600 11px ${font}`;
	g.fillText(signed(cp, 2), node[0], node[1]);
	g.font = `500 11px ${font}`;
	g.fillStyle = '#58727f';
	g.textBaseline = 'alphabetic';
	layers.forEach((values, l) => g.fillText(String(values.length), xs[l] + STRIP / 2, HEIGHT - 6));
	g.fillText('Cp', node[0], HEIGHT - 6);
	g.textBaseline = 'top';
	for (let l = 0; l < xs.length - 1; l++) g.fillText(card.activation, (xs[l + 1] + xs[l] + STRIP) / 2, 2);
}

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./groups.js').InputGroup[] }} options
 */
export function createNetworks({ card, groups }) {
	const canvas = h('canvas', { class: 'network-canvas', role: 'img', 'aria-label': 'One network working on the point' });
	const state = { network: 0, point: null };
	const runs = card.trainRuns;
	const cpCells = card.networks.map(() => h('span', { class: 'fold-cp num' }));
	const foldRows = card.networks.map((network, k) => h('div', {
		class: 'fold-row', role: 'radio', tabindex: 0, 'aria-checked': String(k === 0),
		onclick: () => choose(k),
		onkeydown: (e) => {
			if (e.key !== 'Enter' && e.key !== ' ') return;
			e.preventDefault();
			choose(k);
		},
	},
	h('span', { class: 'fold-name' }, `Network ${k + 1}`),
	runs.map((run) => h('i', { class: `fold-car${network.validationRuns.includes(run) ? ' held' : ''}`, title: `Run ${run}` })),
	cpCells[k]));

	const sameInputs = card.comparison.filter((row) => row.learner !== 'constant' && groupsGiven(groups, row).length === groups.length);
	const network = card.networks[0];
	const trainedOn = network.trainRuns.length;
	const heldOut = network.validationRuns.length;

	const el = section('networks', `${countWord(card.networks.length)} small networks, each blind to ${countWord(heldOut).toLowerCase()} whole cars`,
		`Each is trained on ${countWord(trainedOn).toLowerCase()} cars and graded on the ${countWord(heldOut).toLowerCase()} it never saw. Cells of one car look alike, so a random split of cells would grade the model on cars it has already seen.`,
		h('div', { class: 'networks-layout' },
			h('figure', { class: 'figure' },
				h('figcaption', {}, 'One network at the point'),
				h('div', { class: 'network-holder' }, canvas),
				h('p', { class: 'small-number num' }, `${network.layers.join(' → ')} · ${card.activation} · ${count(parameters(network.layers))} weights`)),
			h('figure', { class: 'figure' },
				h('figcaption', {}, 'Cars each network trained on'),
				h('div', { class: 'folds', role: 'radiogroup', 'aria-label': 'Networks' },
					h('div', { class: 'fold-row fold-head' }, h('span'), runs.map((run) => h('span', { class: 'fold-run num' }, run)), h('span', { class: 'fold-cp' }, 'Cp here')),
					foldRows),
				h('p', { class: 'fold-key' }, h('span', {}, h('i', { class: 'fold-car' }), 'trained on'), h('span', {}, h('i', { class: 'fold-car held' }), 'held out, graded on')),
				h('figcaption', {}, 'Same inputs, other learners, cross-validated R²'),
				bars(sameInputs.map((row) => ({
					label: learnerName(row, groups), chips: chips(groups, groupsGiven(groups, row)), value: row.crossValidationR2,
					text: row.crossValidationR2.toFixed(3), strong: row.learner === 'ensemble',
				})), 1))));
	el.prepend(locator('networks'));

	function draw() {
		const explanation = state.point?.explanation;
		if (!explanation) return;
		drawNetwork(canvas, { card, groups, layers: explanation.activations[state.network], cp: explanation.cpModels[state.network] });
		explanation.cpModels.forEach((cp, k) => (cpCells[k].textContent = signed(cp, 3)));
	}

	function choose(k) {
		state.network = k;
		foldRows.forEach((row, i) => row.setAttribute('aria-checked', String(i === k)));
		draw();
	}

	new ResizeObserver(draw).observe(canvas.parentElement);

	return {
		el,
		setCase() {},
		setPoint(point) {
			state.point = point;
			draw();
		},
	};
}
