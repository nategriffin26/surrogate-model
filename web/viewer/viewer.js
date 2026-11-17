// The 3D view: one car on a studio floor, coloured through a ramp in the shader, drawn only when something changes.
// It draws the file's full mesh whenever the device can, and the working mesh while the camera moves on devices
// where measured frame times say the full mesh cannot keep up. Picking, probes and cuts use the working mesh.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { rampRow, rampTexture } from '../app/ramps.js';
import { Ground, Lattice } from './ground.js';
import { VertexGrid } from './picking.js';
import { CAR_FRAGMENT, CAR_VERTEX, EDGE_FRAGMENT, EDGE_VERTEX, PICK_FRAGMENT, PICK_VERTEX } from './shaders.js';

const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)');
const SHADER_FIELD = { cp: 0, cfd: 1, error: 2, sigma: 3, drag: 4, none: 5 };
const INK = '#0d2233';
const CLAY = '#e4e9ed';
const ACCENT = '#0b6fbf';
const FOV = 30;
const ORTHO_DISTANCE = 20;
/** The camera counts as moving until this long after its last change. */
const SETTLE_MS = 140;
/** Screen motion per frame, in pixels, that counts as the camera moving (for choosing the mesh). */
const MOVING_PX = 0.5;
/** Screen motion per frame, in pixels, below which a frame is not worth drawing. */
const SUBPIXEL = 0.05;
/** A full-mesh frame slower than this while the camera moves hands motion to the working mesh. */
const MOVING_BUDGET_MS = 14;
const ARRIVAL_MS = 850;

/** Camera directions, from the target towards the camera. x runs downstream, z up, the car's left side is −y. */
export const VIEWS = {
	isometric: { label: 'Isometric', dir: [-1, -1, 0.9] },
	front: { label: 'Front', dir: [-1, 0, 0] },
	rear: { label: 'Rear', dir: [1, 0, 0] },
	left: { label: 'Left', dir: [0, -1, 0] },
	right: { label: 'Right', dir: [0, 1, 0] },
	top: { label: 'Top', dir: [0, -1e-4, 1] },
	bottom: { label: 'Bottom', dir: [0, -1e-4, -1] },
};

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export class Viewer extends EventTarget {
	constructor(host) {
		super();
		this.host = host;
		this.surface = null;
		this.bounds = null;
		this.panes = [{ field: 'none', map: null }];
		this.display = { edges: false, ground: true };
		this.section = null;
		this.markers = null;
		this.animations = new Set();
		this.dirty = true;
		this.pending = false;
		this.lastMotion = -Infinity;
		this.lastFrame = 0;
		this.drawn = null;
		this.lod = { budget: MOVING_BUDGET_MS };
		this.resetLod(null);

		const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
		renderer.setClearColor(0x000000, 0);
		renderer.outputColorSpace = THREE.SRGBColorSpace;
		renderer.domElement.className = 'viewer-canvas';
		host.append(renderer.domElement);
		this.renderer = renderer;
		this.canvas = renderer.domElement;
		// three rebuilds its own state when the GPU hands the context back; the floor shadow lived in a texture.
		this.canvas.addEventListener('webglcontextrestored', () => {
			if (this.surface) this.placeSurface();
			this.requestRender();
		});

		this.perspective = new THREE.PerspectiveCamera(FOV, 1, 0.05, 100);
		this.orthographic = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.05, ORTHO_DISTANCE * 3);
		for (const camera of [this.perspective, this.orthographic]) camera.up.set(0, 0, 1);
		this.camera = this.perspective;
		this.camera.position.set(-6, -6, 5);

		const controls = new OrbitControls(this.camera, this.canvas);
		controls.enableDamping = true;
		controls.dampingFactor = 0.14;
		controls.zoomToCursor = true;
		controls.screenSpacePanning = true;
		controls.rotateSpeed = 0.8;
		controls.zoomSpeed = 1.1;
		controls.minDistance = 0.05;
		controls.maxDistance = 40;
		controls.minZoom = 1;
		controls.maxZoom = 5000;
		controls.addEventListener('start', () => {
			this.cancelAnimation('camera');
			this.canvas.classList.add('grabbing');
		});
		controls.addEventListener('end', () => this.canvas.classList.remove('grabbing'));
		controls.addEventListener('change', () => this.schedule());
		this.controls = controls;

		this.uniforms = {
			uRamp: { value: rampTexture() },
			uField: { value: SHADER_FIELD.none },
			uMap: { value: new THREE.Vector4(-1, 1, 0, 0) },
			uClay: { value: new THREE.Color(CLAY) },
			uInk: { value: new THREE.Color(INK) },
			uKey: { value: new THREE.Vector3(-0.45, -0.55, 0.7).normalize() },
			uFill: { value: new THREE.Vector3(0.7, 0.5, 0.2).normalize() },
			uShade: { value: 0.8 },
			uGloss: { value: 0.35 },
			uZmin: { value: 0 },
			uOrtho: { value: 0 },
			uViewDir: { value: new THREE.Vector3() },
			uSection: { value: 0 },
			uPlane: { value: new THREE.Vector4(0, 1, 0, 0) },
			uClip: { value: 0 },
			uArrival: { value: new THREE.Vector4() },
			uFade: { value: 1 },
		};
		this.carMaterial = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: CAR_VERTEX, fragmentShader: CAR_FRAGMENT, side: THREE.DoubleSide });
		this.car = new THREE.Mesh(new THREE.BufferGeometry(), this.carMaterial);
		this.car.frustumCulled = false;
		this.car.visible = false;

		this.edgeMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uInk: this.uniforms.uInk,
				uPlane: this.uniforms.uPlane,
				uClip: this.uniforms.uClip,
				uOrtho: this.uniforms.uOrtho,
				uPixel: { value: 1 },
			},
			vertexShader: EDGE_VERTEX,
			fragmentShader: EDGE_FRAGMENT,
			transparent: true,
			depthWrite: false,
		});
		this.edges = new THREE.LineSegments(new THREE.BufferGeometry(), this.edgeMaterial);
		this.edges.frustumCulled = false;
		this.edges.visible = false;
		this.edges.renderOrder = 2;

		this.ground = new Ground(this.uniforms);
		this.lattice = new Lattice();
		this.lattice.object.visible = false;
		this.backdrop = 'none';

		this.plane = this.buildPlane();

		this.scene = new THREE.Scene();
		this.scene.add(this.car, this.edges, this.ground.object, this.lattice.object, this.plane);

		this.pickMaterial = new THREE.ShaderMaterial({
			uniforms: { uPlane: this.uniforms.uPlane, uClip: this.uniforms.uClip },
			vertexShader: PICK_VERTEX,
			fragmentShader: PICK_FRAGMENT,
			side: THREE.DoubleSide,
		});
		this.pickMesh = new THREE.Mesh(new THREE.BufferGeometry(), this.pickMaterial);
		this.pickMesh.frustumCulled = false;
		this.pickScene = new THREE.Scene();
		this.pickScene.add(this.pickMesh);
		this.pickTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });
		this.grid = null;
		/** Whether pointer moves are picked for a hover readout; the app turns it on once a case has a result. */
		this.hoverPicking = false;
		this.pointer = null;
		this.hoverNext = null;
		this.hoverBusy = false;

		this.bindPointer();
		this.resizeObserver = new ResizeObserver(() => this.resize());
		this.resizeObserver.observe(host);
		this.resize();
		this.warmUp();
	}

	/**
	 * Link every shader program and query the extension three asks for lazily, now, while the GPU is idle. A program's
	 * first use and that query each wait on the GPU process, and their first use comes mid-analysis, when the GPU
	 * process is busy running the networks: the shadow bake alone stalled the page 3.6 s that way.
	 */
	warmUp() {
		const renderer = this.renderer;
		renderer.extensions.has('EXT_texture_filter_anisotropic');
		const speck = new THREE.BufferGeometry();
		speck.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
		speck.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(9), 3));
		const objects = [];
		this.scene.traverse((o) => objects.push({ o, visible: o.visible, culled: o.frustumCulled }));
		for (const { o } of objects) Object.assign(o, { visible: true, frustumCulled: false });
		const geometries = [this.car.geometry, this.edges.geometry, this.pickMesh.geometry];
		this.car.geometry = this.edges.geometry = this.pickMesh.geometry = speck;
		renderer.render(this.scene, this.camera);
		renderer.setRenderTarget(this.pickTarget);
		renderer.render(this.pickScene, this.camera);
		renderer.setRenderTarget(null);
		this.ground.bakeShadow(renderer, speck, [0, 1, 0, 1, 0]);
		[this.car.geometry, this.edges.geometry, this.pickMesh.geometry] = geometries;
		for (const { o, visible, culled } of objects) Object.assign(o, { visible, frustumCulled: culled });
		renderer.clear();
		speck.dispose();
	}

	// ---- the car ------------------------------------------------------------------------------------------------

	/** Show a case's surface, or nothing (the first screen). The floor is laid under the working mesh. */
	show(surface) {
		this.surface = surface;
		this.bounds = null;
		this.grid = null;
		this.car.visible = surface !== null;
		this.ground.object.visible = surface !== null && this.display.ground;
		if (surface) this.placeSurface();
		this.requestRender();
	}

	/** The surface gained a mesh or fields: re-lay the floor if the working mesh changed, and redraw. */
	surfaceChanged() {
		if (this.pickMesh.geometry !== this.surface.working.geometry) this.placeSurface();
		this.requestRender();
	}

	placeSurface() {
		const working = this.surface.working;
		this.bounds = working.bounds;
		this.pickMesh.geometry = working.geometry;
		this.grid = null;
		this.uniforms.uZmin.value = this.bounds[4];
		this.ground.place(this.renderer, working.geometry, this.bounds);
	}

	/** The vertex grid is built on first use: picking is the only thing that needs it. */
	vertexGrid() {
		this.grid ??= new VertexGrid(this.surface.working.mesh.positions);
		return this.grid;
	}

	/**
	 * Which mesh to draw now: the full mesh when it carries every field shown, unless motion must go faster. While the
	 * networks are still predicting the full mesh they share the GPU with drawing, so motion stays on the working mesh
	 * and its cost on the full mesh is timed only once they finish.
	 */
	chooseLayer(moving) {
		if (!this.surface) return null;
		const full = this.surface.full;
		// The full mesh shown as the preview is, until the result arrives, the working mesh too.
		if (!full || full === this.surface.working) return 'working';
		if (!this.panes.every((pane) => full.has(pane.field))) return 'working';
		if (this.lod.layer !== full) this.resetLod(full);
		if (moving && (this.surface.fullFraction < 1 || !this.lod.fullWhileMoving)) return 'working';
		return 'full';
	}

	/**
	 * Panes side by side, each with a field and its colour map {ramp, lo, hi, bands}. One pane is the normal view;
	 * two share one camera for comparison.
	 */
	setPanes(panes) {
		const count = this.panes.length;
		this.panes = panes;
		if (panes.length !== count) this.resize();
		this.requestRender();
	}

	setDisplay(display) {
		this.display = { ...this.display, ...display };
		this.ground.object.visible = this.surface !== null && this.display.ground;
		this.requestRender();
	}

	/** A section plane normal to an axis: {axis: 0|1|2, offset, clip: -1|0|1, plane: whether to draw the plane}, or null. */
	setSection(section) {
		const u = this.uniforms;
		const before = u.uPlane.value.toArray().concat(u.uClip.value);
		this.section = section;
		u.uSection.value = section ? 1 : 0;
		u.uClip.value = section ? section.clip : 0;
		if (section) {
			const n = [0, 0, 0];
			n[section.axis] = 1;
			u.uPlane.value.set(...n, section.offset);
		}
		// The floor shadow follows the clipped car.
		const after = u.uPlane.value.toArray().concat(u.uClip.value);
		if (this.surface && (before[4] !== 0 || after[4] !== 0) && before.some((v, i) => v !== after[i])) {
			this.ground.bakeShadow(this.renderer, this.surface.working.geometry, this.bounds);
		}
		this.placePlane();
		this.requestRender();
	}

	/** Whether a world point lies in the part of the car the section clips away. */
	clipped(point) {
		const s = this.section;
		return Boolean(s?.clip) && s.clip * (point[s.axis] - s.offset) > 0;
	}

	/** Objects drawn with the car in world space (the model view's stencil), or null. The caller owns and disposes them. */
	setMarkers(object) {
		if (this.markers) this.scene.remove(this.markers);
		this.markers = object;
		if (object) this.scene.add(object);
		this.requestRender();
	}

	buildPlane() {
		const group = new THREE.Group();
		const fill = new THREE.Mesh(
			new THREE.PlaneGeometry(1, 1),
			new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }),
		);
		const outline = new THREE.LineLoop(
			new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, -0.5, 0), new THREE.Vector3(0.5, 0.5, 0), new THREE.Vector3(-0.5, 0.5, 0)]),
			new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.8 }),
		);
		fill.renderOrder = outline.renderOrder = 3;
		group.add(fill, outline);
		group.visible = false;
		return group;
	}

	placePlane() {
		const s = this.section;
		this.plane.visible = Boolean(s?.plane && this.bounds);
		if (!this.plane.visible) return;
		const b = this.bounds;
		const centre = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
		const size = [b[1] - b[0], b[3] - b[2], b[5] - b[4]].map((d) => d * 1.08 + 0.1);
		centre[s.axis] = s.offset;
		this.plane.position.set(...centre);
		// The plane geometry lies in its local xy; turn it so its normal is the section axis.
		const [u, v] = [[1, 2], [0, 2], [0, 1]][s.axis];
		this.plane.rotation.set(0, 0, 0);
		if (s.axis === 0) this.plane.rotation.set(Math.PI / 2, Math.PI / 2, 0);
		if (s.axis === 1) this.plane.rotation.set(Math.PI / 2, 0, 0);
		this.plane.scale.set(size[u], size[v], 1);
	}

	// ---- the floor and the lattice --------------------------------------------------------------------------------

	/** 'still' or 'moving' lattice before a car is shown; 'none' fades it out. */
	setBackdrop(mode) {
		const lattice = this.lattice;
		const u = lattice.uniforms;
		this.backdrop = mode;
		switch (mode) {
			case 'none':
				if (!lattice.object.visible) return;
				this.animate(700, (k) => {
					u.uOpacity.value = 1 - k;
					u.uAmp.value = 0.18 * (1 - k);
				}, () => (lattice.object.visible = false), 'backdrop');
				return;
			case 'still':
			case 'moving':
				this.cancelAnimation('backdrop');
				lattice.object.visible = true;
				u.uOpacity.value = 1;
				u.uAmp.value = 0.18;
				this.requestRender();
				return;
			default:
				throw new Error(`Unknown backdrop ${mode}`);
		}
	}

	/** Position, 0 to 1, of the band that crosses the lattice while a file is analysed; null hides it. */
	setSweep(k) {
		this.lattice.uniforms.uSweep.value = k === null ? -100 : -5 + 14.5 * k;
		this.requestRender();
	}

	/** The field arrives on the car as a front carried downstream, broken into streaks like dye in a tunnel. */
	arrive() {
		const u = this.uniforms;
		if (REDUCED_MOTION.matches) {
			this.animate(200, (k) => (u.uFade.value = k), () => (u.uFade.value = 1), 'arrival');
			return;
		}
		const [xmin, xmax] = this.bounds;
		const length = xmax - xmin;
		const width = 0.22 * length;
		const depth = 0.1 * length;
		u.uArrival.value.set(xmin - depth, width, depth, 1);
		this.animate(ARRIVAL_MS, (k) => (u.uArrival.value.x = xmin - depth + (length + width + depth) * (k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2)), () => (u.uArrival.value.w = 0), 'arrival');
	}

	// ---- camera --------------------------------------------------------------------------------------------------

	get target() {
		return this.controls.target;
	}

	/** The camera placement that frames the car looking along `dir`; returned, not applied. */
	framing(dir, camera = this.camera) {
		const b = this.bounds ?? [-0.77, 3.94, -1.01, 1.01, -0.32, 1.2];
		const target = new THREE.Vector3((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2);
		const d = new THREE.Vector3(...dir).normalize();
		const corners = [];
		for (const x of [b[0], b[1]]) for (const y of [b[2], b[3]]) for (const z of [b[4], b[5]]) corners.push(new THREE.Vector3(x, y, z));
		const probe = camera.clone();
		probe.up.set(0, 0, 1);
		const fill = 0.9;
		const { w, h } = this.paneSize();
		if (camera.isOrthographicCamera) {
			probe.position.copy(target).addScaledVector(d, ORTHO_DISTANCE);
			probe.lookAt(target);
			probe.updateMatrixWorld();
			let extent = 0;
			for (const p of corners) {
				const q = p.clone().applyMatrix4(probe.matrixWorldInverse);
				extent = Math.max(extent, Math.abs(q.x) / (w / 2), Math.abs(q.y) / (h / 2));
			}
			return { position: probe.position.clone(), target, zoom: fill / extent };
		}
		const tanY = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
		const tanX = tanY * (w / h);
		let distance = 10;
		for (let i = 0; i < 4; i++) {
			probe.position.copy(target).addScaledVector(d, distance);
			probe.lookAt(target);
			probe.updateMatrixWorld();
			let extent = 0;
			for (const p of corners) {
				const q = p.clone().applyMatrix4(probe.matrixWorldInverse);
				extent = Math.max(extent, Math.abs(q.x) / (-q.z * tanX), Math.abs(q.y) / (-q.z * tanY));
			}
			distance *= extent / fill;
		}
		return { position: target.clone().addScaledVector(d, distance), target, zoom: 1 };
	}

	/** Go to a standard view, animated along an arc unless motion is reduced. */
	view(name, { animate = true } = {}) {
		this.moveTo(this.framing(VIEWS[name].dir), animate);
	}

	/** Look at the car along a direction (the axis triad). */
	lookAlong(dir) {
		this.moveTo(this.framing(dir), true);
	}

	/** Frame the whole car without changing the direction of view. */
	fit() {
		this.moveTo(this.framing(this.camera.position.clone().sub(this.target).toArray()), true);
	}

	/** Make a surface point the orbit centre, keeping the direction and distance of view. */
	focus(point) {
		const to = new THREE.Vector3(...point);
		const offset = this.camera.position.clone().sub(this.target);
		this.moveTo({ position: to.clone().add(offset), target: to, zoom: this.camera.zoom }, true, 350);
	}

	moveTo(goal, animated, ms = 450) {
		this.cancelAnimation('camera');
		this.stopGlide();
		const camera = this.camera;
		if (!animated || REDUCED_MOTION.matches) {
			camera.position.copy(goal.position);
			this.target.copy(goal.target);
			camera.zoom = goal.zoom;
			camera.updateProjectionMatrix();
			this.controls.update();
			this.requestRender();
			return;
		}
		// Interpolate on a sphere about the moving target so the camera swings around the car rather than through it.
		const fromTarget = this.target.clone();
		const toYUp = (v) => new THREE.Vector3(v.x, v.z, -v.y);
		const fromYUp = (v) => new THREE.Vector3(v.x, -v.z, v.y);
		const s0 = new THREE.Spherical().setFromVector3(toYUp(camera.position.clone().sub(fromTarget)));
		const s1 = new THREE.Spherical().setFromVector3(toYUp(goal.position.clone().sub(goal.target)));
		let dTheta = s1.theta - s0.theta;
		if (dTheta > Math.PI) dTheta -= 2 * Math.PI;
		if (dTheta < -Math.PI) dTheta += 2 * Math.PI;
		const zoom0 = camera.zoom;
		this.animate(ms, (k) => {
			const e = ease(k);
			const s = new THREE.Spherical(s0.radius + (s1.radius - s0.radius) * e, s0.phi + (s1.phi - s0.phi) * e, s0.theta + dTheta * e);
			this.target.copy(fromTarget).lerp(goal.target, e);
			camera.position.copy(this.target).add(fromYUp(new THREE.Vector3().setFromSpherical(s)));
			camera.zoom = zoom0 + (goal.zoom - zoom0) * e;
			camera.updateProjectionMatrix();
			camera.lookAt(this.target);
		}, () => this.controls.update(), 'camera');
	}

	get projection() {
		return this.camera.isOrthographicCamera ? 'orthographic' : 'perspective';
	}

	/** Switch projection keeping what is seen: the car keeps its size at the orbit centre. */
	setProjection(projection) {
		if (projection === this.projection) return;
		this.cancelAnimation('camera');
		this.stopGlide();
		const from = this.camera;
		const dir = from.position.clone().sub(this.target).normalize();
		const tan = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
		const { h } = this.paneSize();
		let to;
		if (projection === 'orthographic') {
			const visible = 2 * from.position.distanceTo(this.target) * tan;
			to = this.orthographic;
			to.position.copy(this.target).addScaledVector(dir, ORTHO_DISTANCE);
			to.zoom = h / visible;
		} else {
			const visible = h / from.zoom;
			to = this.perspective;
			to.position.copy(this.target).addScaledVector(dir, visible / (2 * tan));
			to.zoom = 1;
		}
		this.camera = to;
		this.controls.object = to;
		this.uniforms.uOrtho.value = projection === 'orthographic' ? 1 : 0;
		this.resize();
		this.controls.update();
		this.dispatchEvent(new CustomEvent('projection'));
	}

	/** Everything needed to put the camera back where it was. */
	cameraState() {
		return { projection: this.projection, position: this.camera.position.toArray(), target: this.target.toArray(), zoom: this.camera.zoom };
	}

	restoreCamera(state) {
		this.cancelAnimation('camera');
		this.stopGlide();
		if (state.projection !== this.projection) {
			this.camera = state.projection === 'orthographic' ? this.orthographic : this.perspective;
			this.controls.object = this.camera;
			this.uniforms.uOrtho.value = state.projection === 'orthographic' ? 1 : 0;
			this.resize();
			this.dispatchEvent(new CustomEvent('projection'));
		}
		this.camera.position.fromArray(state.position);
		this.target.fromArray(state.target);
		this.camera.zoom = state.zoom;
		this.camera.updateProjectionMatrix();
		this.controls.update();
		this.requestRender();
	}

	/**
	 * End the controls' damped glide at once. Without this its leftover, too small to see, is applied by the next
	 * update and nudges a camera placed since (a standard view, a restored case) off where it was put.
	 */
	stopGlide() {
		this.controls.enableDamping = false;
		this.controls.update();
		this.controls.enableDamping = true;
	}

	// ---- geometry of the screen ------------------------------------------------------------------------------------

	paneSize() {
		const W = this.host.clientWidth || 1;
		const H = this.host.clientHeight || 1;
		return { W, H, w: W / this.panes.length, h: H };
	}

	/** Pane rectangles in host pixels. */
	paneRects() {
		const { w, h } = this.paneSize();
		return this.panes.map((_, i) => ({ x: i * w, y: 0, w, h }));
	}

	/** Host-relative screen position of a world point in a pane, and whether it is in front of the camera. */
	project(point, pane = 0) {
		const { w, h } = this.paneSize();
		const v = new THREE.Vector3(...point).project(this.camera);
		return { x: pane * w + (v.x * 0.5 + 0.5) * w, y: (-v.y * 0.5 + 0.5) * h, visible: v.z > -1 && v.z < 1 };
	}

	/** The pane under a client point and the point's position inside it. */
	paneAt(clientX, clientY) {
		const rect = this.canvas.getBoundingClientRect();
		const { w, h } = this.paneSize();
		const x = clientX - rect.left;
		const y = clientY - rect.top;
		if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return null;
		const index = Math.min(this.panes.length - 1, Math.floor(x / w));
		return { index, x: x - index * w, y, w, h };
	}

	/** World-space ray through a client point. */
	rayAt(clientX, clientY) {
		const at = this.paneAt(clientX, clientY);
		if (!at) return null;
		const raycaster = new THREE.Raycaster();
		raycaster.setFromCamera(new THREE.Vector2((at.x / at.w) * 2 - 1, -(at.y / at.h) * 2 + 1), this.camera);
		return raycaster.ray;
	}

	resize() {
		const { W, H, w, h } = this.paneSize();
		this.renderer.setSize(W, H, false);
		this.perspective.aspect = w / h;
		this.perspective.updateProjectionMatrix();
		const o = this.orthographic;
		o.left = -w / 2;
		o.right = w / 2;
		o.top = h / 2;
		o.bottom = -h / 2;
		o.updateProjectionMatrix();
		this.resetLod(this.lod.layer);
		this.requestRender();
		this.dispatchEvent(new CustomEvent('resize'));
	}

	// ---- picking ----------------------------------------------------------------------------------------------------

	/**
	 * The working-mesh vertex under a client point: {pane, point (where the cursor meets the surface), vertex}, or
	 * null over the background. Rendered into a one-pixel float target and read back without stalling the page.
	 */
	async pick(clientX, clientY) {
		if (!this.surface) return null;
		const at = this.paneAt(clientX, clientY);
		if (!at) return null;
		const camera = this.camera;
		const renderer = this.renderer;
		camera.setViewOffset(at.w, at.h, at.x, at.y, 1, 1);
		renderer.setRenderTarget(this.pickTarget);
		renderer.setClearColor(0x000000, 0);
		renderer.clear();
		renderer.render(this.pickScene, camera);
		renderer.setRenderTarget(null);
		camera.clearViewOffset();
		const out = new Float32Array(4);
		const surface = this.surface;
		await renderer.readRenderTargetPixelsAsync(this.pickTarget, 0, 0, 1, 1, out);
		if (out[3] === 0 || surface !== this.surface) return null;
		const point = [out[0], out[1], out[2]];
		// The nearest working-mesh vertex needs the vertex grid; it is found only when a reader asks for it.
		const nearest = () => this.vertexGrid().nearest(point);
		let vertex;
		return { pane: at.index, point, get vertex() { return (vertex ??= nearest()); } };
	}

	bindPointer() {
		const el = this.canvas;
		let down = null;
		let tapTimer = 0;
		el.addEventListener('pointerdown', (e) => {
			down = { x: e.clientX, y: e.clientY, button: e.button };
		});
		// A single click waits for a possible second one, so a double-click (pivot) never also counts as a click.
		el.addEventListener('click', (e) => {
			if (!down || down.button !== 0 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
			clearTimeout(tapTimer);
			if (e.detail > 1) return;
			const { clientX, clientY } = e;
			tapTimer = setTimeout(async () => {
				const hit = await this.pick(clientX, clientY);
				this.dispatchEvent(new CustomEvent('tap', { detail: { hit, clientX, clientY } }));
			}, 230);
		});
		el.addEventListener('dblclick', async (e) => {
			clearTimeout(tapTimer);
			const hit = await this.pick(e.clientX, e.clientY);
			if (hit) this.focus(hit.point);
		});
		el.addEventListener('pointermove', (e) => {
			if (!this.hoverPicking || e.buttons !== 0 || e.pointerType === 'touch') return;
			this.pointer = { clientX: e.clientX, clientY: e.clientY };
			this.hover(this.pointer);
		});
		const leave = () => {
			this.pointer = null;
			this.hoverNext = null;
			this.dispatchEvent(new CustomEvent('hover', { detail: { hit: null } }));
		};
		el.addEventListener('pointerleave', leave);
		el.addEventListener('pointerdown', leave);
	}

	/**
	 * Pick under the pointer for the hover readout. A pick takes a frame or two to read back; positions that arrive
	 * meanwhile collapse into the latest. The frame loop also calls this while the camera moves under a still pointer,
	 * so the readout always describes the surface the pointer is over.
	 */
	async hover(position) {
		this.hoverNext = position;
		if (this.hoverBusy) return;
		this.hoverBusy = true;
		while (this.hoverNext) {
			const { clientX, clientY } = this.hoverNext;
			this.hoverNext = null;
			const hit = await this.pick(clientX, clientY);
			if (!this.pointer) break;
			this.dispatchEvent(new CustomEvent('hover', { detail: { hit, clientX, clientY } }));
		}
		this.hoverBusy = false;
	}

	// ---- the frame loop -----------------------------------------------------------------------------------------------

	animate(ms, step, done, key) {
		if (key) this.cancelAnimation(key);
		const animation = { start: performance.now(), ms, step, done, key };
		this.animations.add(animation);
		this.requestRender();
		return animation;
	}

	cancelAnimation(key) {
		for (const a of this.animations) if (a.key === key) this.animations.delete(a);
	}

	requestRender() {
		this.dirty = true;
		this.schedule();
	}

	schedule() {
		if (this.pending || this.disposed) return;
		this.pending = true;
		requestAnimationFrame((now) => this.frame(now));
	}

	frame(now) {
		this.pending = false;
		const animating = this.animations.size > 0;
		for (const a of [...this.animations]) {
			const k = Math.min(1, (now - a.start) / a.ms);
			a.step(k);
			if (k >= 1) {
				this.animations.delete(a);
				a.done?.();
			}
		}
		const cameraAnimating = [...this.animations].some((a) => a.key === 'camera');
		const controlled = !cameraAnimating && this.controls.update();
		// Damping glides on for a second in sub-pixel steps; only motion a viewer can see counts as moving. The camera
		// can also move between frames (a wheel step updates it at once), so motion is measured on screen. The glide is
		// ended once it is invisible, or once the controls stop reporting it (under a millimetre a step), since what
		// is left of it would otherwise be applied by the next update.
		const shift = this.screenMotion();
		if (!cameraAnimating && (!controlled || shift <= SUBPIXEL)) this.stopGlide();
		if (cameraAnimating || shift > MOVING_PX) this.lastMotion = now;
		if (shift > SUBPIXEL && this.pointer && this.hoverPicking) this.hover(this.pointer);
		const latticeMoving = this.lattice.object.visible && this.backdrop === 'moving';
		if (latticeMoving) this.lattice.uniforms.uTime.value += Math.min(100, now - this.lastFrame) / 1000;
		const moving = now - this.lastMotion < SETTLE_MS;
		const layer = this.chooseLayer(moving);
		if (this.dirty || animating || shift > SUBPIXEL || latticeMoving || layer !== this.drawn?.layer) {
			const start = performance.now();
			this.render(layer);
			if (layer === 'full' && moving && !this.lod.decided) this.measure(start);
		}
		this.lastFrame = now;
		this.dirty = false;
		if (this.animations.size > 0 || controlled || moving || latticeMoving) this.schedule();
	}

	/** The largest distance, in pixels, any corner of the car's box moved on screen since the previous frame. */
	screenMotion() {
		const b = this.bounds ?? [-1, 1, -1, 1, -1, 1];
		this.camera.updateMatrixWorld();
		const corners = [];
		for (const x of [b[0], b[1]]) for (const y of [b[2], b[3]]) for (const z of [b[4], b[5]]) corners.push(this.project([x, y, z]));
		const previous = this.corners;
		this.corners = corners;
		if (!previous) return Infinity;
		return Math.max(...corners.map((p, i) => Math.hypot(p.x - previous[i].x, p.y - previous[i].y)));
	}

	/**
	 * Time the first few full-mesh frames of motion to completion on the GPU (a one-pixel read waits for it), then
	 * decide once whether motion can stay on the full mesh. Resizing or a new full mesh asks again.
	 */
	measure(start) {
		const gl = this.renderer.getContext();
		gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
		const lod = this.lod;
		lod.samples.push(performance.now() - start);
		if (lod.samples.length < 5) return;
		lod.decided = true;
		lod.frameMs = median(lod.samples);
		lod.fullWhileMoving = lod.frameMs <= lod.budget;
	}

	/** Forget the measured frame cost: a new full mesh or a new canvas size is measured afresh. */
	resetLod(layer) {
		Object.assign(this.lod, { layer, samples: [], frameMs: null, decided: false, fullWhileMoving: true });
	}

	render(layerName = this.chooseLayer(false)) {
		const renderer = this.renderer;
		const camera = this.camera;
		const u = this.uniforms;
		if (camera.isPerspectiveCamera) {
			const distance = camera.position.distanceTo(this.target);
			camera.near = Math.min(0.5, Math.max(0.002, distance * 0.01));
			camera.far = distance + 40;
			camera.updateProjectionMatrix();
		}
		camera.getWorldDirection(u.uViewDir.value);
		const layer = layerName ? this.surface[layerName] : null;
		if (layer) {
			this.car.geometry = layer.geometry;
			const edgeGeometry = this.display.edges ? layer.edgeGeometry(() => this.requestRender()) : null;
			this.edges.visible = edgeGeometry !== null;
			if (edgeGeometry) this.edges.geometry = edgeGeometry;
			this.carMaterial.polygonOffset = this.edges.visible;
			this.carMaterial.polygonOffsetFactor = 1;
			this.carMaterial.polygonOffsetUnits = 1;
			const { h } = this.paneSize();
			this.edgeMaterial.uniforms.uPixel.value = camera.isOrthographicCamera
				? (camera.top - camera.bottom) / camera.zoom / h
				: (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / h;
		} else {
			this.edges.visible = false;
		}
		const rects = this.paneRects();
		renderer.setScissorTest(rects.length > 1);
		renderer.setViewport(0, 0, this.host.clientWidth, this.host.clientHeight);
		renderer.setScissor(0, 0, this.host.clientWidth, this.host.clientHeight);
		renderer.clear();
		renderer.autoClear = false;
		this.panes.forEach((pane, i) => {
			const r = rects[i];
			renderer.setViewport(r.x, 0, r.w, r.h);
			renderer.setScissor(r.x, 0, r.w, r.h);
			u.uField.value = SHADER_FIELD[pane.field];
			if (pane.map) u.uMap.value.set(pane.map.lo, pane.map.hi, pane.map.bands, rampRow(pane.map.ramp));
			renderer.render(this.scene, camera);
		});
		renderer.autoClear = true;
		renderer.setScissorTest(false);
		// `full` says whether every triangle of the file was drawn: the preview is the full mesh in the working role.
		const drawn = { layer: layer ? layerName : null, triangles: layer ? layer.triangles : 0, full: Boolean(layer) && layer === this.surface.full };
		if (drawn.layer !== this.drawn?.layer || drawn.triangles !== this.drawn?.triangles || drawn.full !== this.drawn?.full) {
			this.drawn = drawn;
			this.dispatchEvent(new CustomEvent('drawn', { detail: drawn }));
		}
		this.dispatchEvent(new CustomEvent('camera'));
	}

	/** Draw now at full detail and return the canvas; read it in the same task, before the browser clears it. */
	snapshot() {
		this.render(this.chooseLayer(false));
		return this.canvas;
	}

	dispose() {
		this.disposed = true;
		this.resizeObserver.disconnect();
		this.controls.dispose();
		this.renderer.dispose();
		this.canvas.remove();
	}
}
