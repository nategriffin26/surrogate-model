// The ten training cars as `export_training_cars.py` wrote them (`web/model/training-cars.json`): each seen from its
// left side as a coarse raster of its CFD pressure, drawn solid or as a dot per cell.

import { h } from '../dom.js';
import { rampPosition, sample } from '../ramps.js';

const TRAINING_CARS = new URL('../../model/training-cars.json', import.meta.url);

/** @returns {Promise<{ step: number, cars: { run: number, cells: number, cdP: number, x0: number, z0: number, nx: number, nz: number, cp: (number | null)[] }[] }>} */
export async function loadTrainingCars() {
	const response = await fetch(TRAINING_CARS);
	if (!response.ok) throw new Error(`The training cars could not be read (${response.status} for ${TRAINING_CARS.pathname}).`);
	return response.json();
}

/** CSS pixels per raster cell when a car is drawn as dots. */
const DOT_CELL = 4;

/**
 * A training car's raster as a canvas, coloured through a colour map {ramp, lo, hi, bands}: 'solid', one pixel per
 * cell, scaled by the page, or 'dots', one dot per cell at DOT_CELL pixels apart, so the car reads as the many cells
 * it is made of.
 */
export function trainingCar({ run, nx, nz, cp }, map, style) {
	const colour = (hundredths) => sample(map.ramp, rampPosition(hundredths / 100, map.lo, map.hi, map.bands));
	if (style === 'solid') {
		const canvas = h('canvas', { class: 'training-car solid', width: nx, height: nz, role: 'img', 'aria-label': `Run ${run}` });
		const g = canvas.getContext('2d');
		const image = g.createImageData(nx, nz);
		cp.forEach((value, i) => {
			if (value === null) return;
			const row = nz - 1 - Math.floor(i / nx);
			image.data.set([...colour(value), 255], 4 * (row * nx + (i % nx)));
		});
		g.putImageData(image, 0, 0);
		return canvas;
	}
	const ratio = devicePixelRatio;
	const canvas = h('canvas', {
		class: 'training-car dots', width: nx * DOT_CELL * ratio, height: nz * DOT_CELL * ratio, role: 'img', 'aria-label': `Run ${run}`,
		style: `width: ${nx * DOT_CELL}px`,
	});
	const g = canvas.getContext('2d');
	g.scale(ratio, ratio);
	cp.forEach((value, i) => {
		if (value === null) return;
		const [r, gr, b] = colour(value);
		g.fillStyle = `rgb(${r}, ${gr}, ${b})`;
		g.beginPath();
		g.arc(((i % nx) + 0.5) * DOT_CELL, (nz - 1 - Math.floor(i / nx) + 0.5) * DOT_CELL, DOT_CELL * 0.34, 0, 2 * Math.PI);
		g.fill();
	});
	return canvas;
}
