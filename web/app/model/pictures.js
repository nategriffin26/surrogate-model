// Pictures of the car coloured by a field, seen square-on from the side or from below, taken from the page's own
// viewer between two of its frames: the camera, the floor and the drawing on the car are put back before the
// browser shows the canvas again, so the live view never shows a picture being taken.

import { h } from '../dom.js';
import { ORTHO_DISTANCE, VIEWS } from '../../viewer/viewer.js';

/** Camera directions, from the car towards the camera: the left side (−y), and from below with the nose to the left. */
const DIRECTIONS = { side: VIEWS.left.dir, below: VIEWS.bottom.dir };
/** Share of the canvas the car fills. */
const FILL = 0.96;

/**
 * Resolves once the viewer draws `surface` with every array on the GPU; with false if it shows another surface first.
 * @param {import('../../viewer/viewer.js').Viewer} viewer
 */
export function whenDrawn(viewer, surface) {
	return new Promise((resolve) => {
		const check = () => {
			const done = viewer.surface !== surface || (viewer.placed === surface.working && !surface.uploading);
			if (!done) return;
			viewer.removeEventListener('camera', check);
			resolve(viewer.surface === surface);
		};
		viewer.addEventListener('camera', check);
		viewer.requestRender();
	});
}

/**
 * @typedef {Object} Picture
 * @property {HTMLCanvasElement} canvas  the car alone, cropped to its outline
 * @property {(point: number[]) => number[]} place  where a world point falls on the picture, as shares of its
 *   width and height from its top left
 */

/**
 * Take pictures of the shown surface. Each request is a field and its colour map {ramp, lo, hi, bands}, or 'none'
 * for clay, and a view.
 * @param {import('../../viewer/viewer.js').Viewer} viewer
 * @param {{ field: string, map: object | null, view: 'side' | 'below' }[]} requests
 * @returns {Picture[]}
 */
export function takePictures(viewer, requests) {
	const camera = viewer.cameraState();
	const panes = viewer.panes;
	const ground = viewer.display.ground;
	const overlay = viewer.overlay;
	if (overlay) overlay.visible = false;
	viewer.setDisplay({ ground: false });
	const pictures = requests.map(({ field, map, view }) => {
		viewer.setPanes([{ field, map }]);
		const target = viewer.bounds;
		frame(viewer, DIRECTIONS[view]);
		viewer.snapshot();
		return crop(viewer, target);
	});
	viewer.setPanes(panes);
	viewer.setDisplay({ ground });
	if (overlay) overlay.visible = true;
	viewer.restoreCamera(camera);
	viewer.render();
	return pictures;
}

/** Point an orthographic camera at the car along `dir`, close enough that the car fills the canvas. */
function frame(viewer, dir) {
	const [x0, x1, y0, y1, z0, z1] = viewer.bounds;
	const target = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
	const extent = dir[2] === -1 ? [x1 - x0, y1 - y0] : [x1 - x0, z1 - z0];
	const { w, h: height } = viewer.paneSize();
	const zoom = FILL * Math.min(w / extent[0], height / extent[1]);
	const length = Math.hypot(...dir);
	viewer.restoreCamera({
		projection: 'orthographic',
		position: target.map((t, k) => t + (dir[k] / length) * ORTHO_DISTANCE),
		target,
		zoom,
	});
}

/** The car's part of the canvas just drawn, and where world points fall on it. */
function crop(viewer, bounds) {
	const ratio = viewer.renderer.getPixelRatio();
	const [x0, x1, y0, y1, z0, z1] = bounds;
	const corners = [];
	for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) corners.push(viewer.project([x, y, z]));
	const left = Math.min(...corners.map((p) => p.x));
	const top = Math.min(...corners.map((p) => p.y));
	const width = Math.max(...corners.map((p) => p.x)) - left;
	const height = Math.max(...corners.map((p) => p.y)) - top;
	const canvas = h('canvas', { width: Math.round(width * ratio), height: Math.round(height * ratio) });
	canvas.getContext('2d').drawImage(viewer.canvas, left * ratio, top * ratio, width * ratio, height * ratio, 0, 0, canvas.width, canvas.height);
	// The camera is orthographic, so a world point's place is affine in it: read the map off three axis steps.
	const origin = viewer.project([0, 0, 0]);
	const axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((a) => viewer.project(a)).map((p) => [p.x - origin.x, p.y - origin.y]);
	const place = (point) => {
		const x = origin.x + point[0] * axes[0][0] + point[1] * axes[1][0] + point[2] * axes[2][0];
		const y = origin.y + point[0] * axes[0][1] + point[1] * axes[1][1] + point[2] * axes[2][1];
		return [(x - left) / width, (y - top) / height];
	};
	return { canvas, place };
}
