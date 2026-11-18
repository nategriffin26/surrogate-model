// Colour ramps. One texture holds every ramp as a row; the shader and the legend both sample it through
// `sample`, so the legend always matches the car.

import * as THREE from 'three';

const turbo = (t) =>
	[
		0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))),
		0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))),
		0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))),
	].map((c) => Math.round(255 * Math.min(1, Math.max(0, c))));

const even = (colours) => colours.map((c, i) => [i / (colours.length - 1), c]);

/** Each ramp is a list of [position, sRGB hex] stops, or a function of t returning sRGB 0–255. */
export const RAMPS = {
	coolwarm: {
		label: 'Cool to warm',
		stops: [
			[0, '#3b4cc0'],
			[0.1, '#4f69d9'],
			[0.2, '#6788ee'],
			[0.3, '#81a4fb'],
			[0.4, '#a9c6fd'],
			[0.5, '#efeeec'],
			[0.6, '#f5c2aa'],
			[0.7, '#f39b7f'],
			[0.8, '#e57058'],
			[0.9, '#cf453c'],
			[1, '#b40426'],
		],
	},
	turbo: { label: 'Turbo', fn: turbo },
	viridis: {
		label: 'Viridis',
		stops: even([
			'#440154',
			'#482475',
			'#414487',
			'#355f8d',
			'#2a788e',
			'#21918c',
			'#22a884',
			'#44bf70',
			'#7ad151',
			'#bddf26',
			'#fde725',
		]),
	},
	purplegreen: {
		label: 'Purple to green',
		stops: even([
			'#40004b',
			'#762a83',
			'#9970ab',
			'#c2a5cf',
			'#e7d4e8',
			'#f7f7f7',
			'#d9f0d3',
			'#a6dba0',
			'#5aae61',
			'#1b7837',
			'#00441b',
		]),
	},
	brownteal: {
		label: 'Brown to teal',
		stops: even([
			'#543005',
			'#8c510a',
			'#bf812d',
			'#dfc27d',
			'#f6e8c3',
			'#f5f5f5',
			'#c7eae5',
			'#80cdc1',
			'#35978f',
			'#01665e',
			'#003c30',
		]),
	},
	purples: { label: 'Purples', stops: even(['#f1eff6', '#d8cbef', '#b38deb', '#8c4fe0', '#4b1e8a']) },
	inferno: {
		label: 'Inferno',
		stops: even([
			'#000004',
			'#160b39',
			'#420a68',
			'#6a176e',
			'#932667',
			'#bc3754',
			'#dd513a',
			'#f37819',
			'#fca50a',
			'#f6d746',
			'#fcffa4',
		]),
	},
	tealorange: {
		label: 'Teal to orange',
		stops: [
			[0, '#0f5f63'],
			[0.25, '#4fa3a6'],
			[0.5, '#efeeec'],
			[0.75, '#e59a55'],
			[1, '#a34a10'],
		],
	},
};

const ROWS = Object.keys(RAMPS);
const WIDTH = 256;

function hexRgb(hex) {
	const n = parseInt(hex.slice(1), 16);
	return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** sRGB 0–255 colour of a ramp at t in [0, 1]. */
export function sample(name, t) {
	const ramp = RAMPS[name];
	const u = Math.min(1, Math.max(0, t));
	if (ramp.fn) return ramp.fn(u);
	const stops = ramp.stops;
	let i = 1;
	while (i < stops.length - 1 && stops[i][0] < u) i++;
	const [t0, c0] = stops[i - 1];
	const [t1, c1] = stops[i];
	const k = (u - t0) / (t1 - t0);
	const a = hexRgb(c0);
	const b = hexRgb(c1);
	return [0, 1, 2].map((j) => Math.round(a[j] + (b[j] - a[j]) * k));
}

/** Position of a value on a ramp spanning [lo, hi], optionally quantised into `bands` equal steps. */
export function rampPosition(value, lo, hi, bands) {
	const t = Math.min(1, Math.max(0, (value - lo) / (hi - lo)));
	if (!bands) return t;
	return (Math.min(bands - 1, Math.floor(t * bands)) + 0.5) / bands;
}

/** CSS gradient of a ramp, for swatches. */
export function rampCss(name, direction = '90deg') {
	const stops = [];
	for (let i = 0; i <= 16; i++) {
		const [r, g, b] = sample(name, i / 16);
		stops.push(`rgb(${r} ${g} ${b}) ${((i / 16) * 100).toFixed(1)}%`);
	}
	return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

export function rampTexture() {
	const data = new Uint8Array(WIDTH * ROWS.length * 4);
	ROWS.forEach((name, row) => {
		for (let i = 0; i < WIDTH; i++) {
			const [r, g, b] = sample(name, i / (WIDTH - 1));
			const o = 4 * (row * WIDTH + i);
			data[o] = r;
			data[o + 1] = g;
			data[o + 2] = b;
			data[o + 3] = 255;
		}
	});
	const texture = new THREE.DataTexture(data, WIDTH, ROWS.length, THREE.RGBAFormat, THREE.UnsignedByteType);
	texture.colorSpace = THREE.SRGBColorSpace;
	texture.magFilter = THREE.LinearFilter;
	texture.minFilter = THREE.LinearFilter;
	texture.needsUpdate = true;
	return texture;
}

/** Texture v-coordinate of a ramp's row. */
export function rampRow(name) {
	return (ROWS.indexOf(name) + 0.5) / ROWS.length;
}
