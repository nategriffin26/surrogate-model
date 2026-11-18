// Line icons on a 16 × 16 grid, drawn with the current text colour.

import { s } from './dom.js';

const PATHS = {
	fit: 'M2 5.5V2h3.5M10.5 2H14v3.5M14 10.5V14h-3.5M5.5 14H2v-3.5M5.5 6.5h5v3h-5z',
	views: 'M8 1.8 13.8 5v6L8 14.2 2.2 11V5zM2.2 5 8 8.2 13.8 5M8 8.2v6',
	edges: 'M2 13.5 6 3l4.5 7.5L14 4M2 13.5h12L10.5 10.5 6 3M2 13.5l4.5-3.5L10.5 10.5M6 3l.5 7',
	ground: 'M1.5 11h13M3.5 13.5h9M6 6.5a2 2 0 1 0 4 0 2 2 0 1 0-4 0',
	section: 'M3 3.5h10v9H3zM8 1.5v13M5.5 6.5 8 5l2.5 1.5',
	compare: 'M1.8 3h12.4v10H1.8zM8 3v10',
	controls: 'M5 2.5h6a1.5 1.5 0 0 1 1.5 1.5v5a4.5 4.5 0 0 1-9 0V4A1.5 1.5 0 0 1 5 2.5zM8 2.5v4M3.5 6.5h9',
	download: 'M8 2v8.5M4.5 7 8 10.5 11.5 7M2.5 13.5h11',
	close: 'M4 4l8 8M12 4l-8 8',
	plus: 'M8 3v10M3 8h10',
	copy: 'M5.5 5.5h7.5v8H5.5zM3 10.5V2.5h7.5',
	reset: 'M3 8a5 5 0 1 0 1.6-3.7M3 2.5V5.5h3',
	symmetric: 'M8 2v12M3 6h3M4.5 4.5v3M10 6h3',
	bands: 'M2 13h3V9.5h3V6h3V2.5h3',
	chevron: 'M4.5 6.5 8 10l3.5-3.5',
	cancel: 'M3.5 3.5h9v9h-9z',
	assistant: 'M2.5 3h11v8H7.5L4.5 13.5V11h-2zM5.5 7h.01M8 7h.01M10.5 7h.01',
	folder: 'M2 4.5a1 1 0 0 1 1-1h3.2l1.5 1.5H13a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z',
	file: 'M4 1.8h5l3 3v9.4H4zM9 1.8v3h3',
	text: 'M4 1.8h5l3 3v9.4H4zM9 1.8v3h3M6 8h4M6 10.5h4',
	geometry: 'M2 13 6 4l4 9zM6 4l6-1.5L10 13M12 2.5l2.5 8.5L10 13',
	image: 'M2.5 3h11v10h-11zM2.5 11 6 7.5l3 3 2-2 2.5 2.5M10.5 5.8h.01',
	table: 'M2.5 3h11v10h-11zM2.5 6.5h11M2.5 10h11M6.5 3v10',
	report: 'M4 1.8h8v12.4H4zM6 5h4M6 7.5h4M6 10h2.5',
	more: 'M3 8h1M7.5 8h1M12 8h1',
	upload: 'M8 10.5V2M4.5 5.5 8 2l3.5 3.5M2.5 13.5h11',
	keep: 'M2 4.5a1 1 0 0 1 1-1h3.2l1.5 1.5H13a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1zM8 7v4M6 9h4',
	run: 'M5 3.5v9l7-4.5z',
	brush: 'M13.5 2.5 7.5 8.5M7.5 8.5l1 1M8.5 9.5c-.5 2-2 3.5-5.5 4 .5-3.5 2-5 4.5-5.5',
	lasso: 'M8 2.5c3.3 0 5.5 1.6 5.5 3.6S11.3 9.7 8 9.7 2.5 8.1 2.5 6.1 4.7 2.5 8 2.5zM5.5 9.3c-.8.9-.8 2.2.4 2.8M5.9 12.1l-1.4 2',
};

export function icon(name) {
	return s('svg', { class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' }, s('path', { d: PATHS[name] }));
}
