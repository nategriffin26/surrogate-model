// The first screen: a drop zone and the bundled examples. Dropping a file anywhere on the window works too.

import { h, s } from './dom.js';
import { EXAMPLES, exampleFacts } from './examples.js';

function meshMark() {
	return s(
		'svg',
		{ viewBox: '0 0 40 28', 'aria-hidden': 'true' },
		s('path', {
			d: 'M2 26 L12 8 L22 26 Z M12 8 L28 4 L22 26 M28 4 L38 22 L22 26',
			fill: 'none',
			stroke: 'currentColor',
			'stroke-width': 1.4,
			'stroke-linejoin': 'round',
		}),
	);
}

function profile(d) {
	return s(
		'svg',
		{ class: 'silhouette', viewBox: '0 0 132 40', 'aria-hidden': 'true' },
		s('line', { x1: 0, x2: 132, y1: 37.5, y2: 37.5, class: 'floor' }),
		s('path', { d }),
	);
}

export function startScreen({ onFile, onExample }) {
	const input = h('input', {
		type: 'file',
		accept: '.vtp',
		hidden: true,
		onchange: () => input.files[0] && onFile(input.files[0]),
	});
	const metas = EXAMPLES.map(() => h('span', { class: 'example-meta' }, '\u00a0'));
	exampleFacts().then((facts) => facts.forEach((text, i) => (metas[i].textContent = text)));
	return h(
		'div',
		{ class: 'start' },
		h(
			'label',
			{ class: 'drop' },
			input,
			h('span', { class: 'drop-mark' }, meshMark()),
			h('span', { class: 'drop-title' }, 'Drop a .vtp file'),
			h('span', { class: 'drop-sub' }, 'or ', h('span', { class: 'link' }, 'choose one')),
		),
		h('div', { class: 'examples-head' }, 'Examples'),
		h(
			'div',
			{ class: 'examples' },
			EXAMPLES.map((example, i) =>
				h(
					'button',
					{ type: 'button', class: 'example', 'data-example': example.id, onclick: () => onExample(example.id) },
					profile(example.profile),
					h('span', { class: 'example-name' }, example.label),
					metas[i],
				),
			),
		),
	);
}

/** Track a drag over the window so the page can show where a drop goes; `onFile` gets the dropped file. */
export function acceptDrops(onFile) {
	let depth = 0;
	const body = document.body;
	const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');
	addEventListener('dragenter', (e) => {
		if (!hasFiles(e)) return;
		e.preventDefault();
		depth++;
		body.classList.add('dragging');
	});
	addEventListener('dragleave', (e) => {
		if (!hasFiles(e)) return;
		depth = Math.max(0, depth - 1);
		if (depth === 0) body.classList.remove('dragging');
	});
	addEventListener('dragover', (e) => hasFiles(e) && e.preventDefault());
	addEventListener('drop', (e) => {
		if (!hasFiles(e)) return;
		e.preventDefault();
		depth = 0;
		body.classList.remove('dragging');
		for (const file of e.dataTransfer.files) onFile(file);
	});
}
