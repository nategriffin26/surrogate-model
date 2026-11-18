// The 3D view: one car on a studio floor, coloured through a ramp in the shader, drawn only when something changes.
// It draws the file's full mesh whenever the device can, and the working mesh while the camera moves on devices
// where measured frame times say the full mesh cannot keep up. Picking, probes and cuts use the working mesh.
// While the networks predict, the clay car shows the work under way, drawn once and tinted each frame (keepStill),
// and the field then arrives on it as a front carried downstream (arrive).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { rampRow, rampTexture } from '../app/ramps.js';
import { Ground, Lattice } from './ground.js';
import { FaceIndex, VertexGrid } from './picking.js';
import { CAR_FRAGMENT, CAR_VERTEX, EDGE_FRAGMENT, EDGE_VERTEX, PICK_FRAGMENT, PICK_VERTEX, SCREEN_VERTEX, WAIT_COMPOSITE_FRAGMENT, WAIT_INFO_FRAGMENT } from './shaders.js';

const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)');
const SHADER_FIELD = { cp: 0, cfd: 1, error: 2, sigma: 3, drag: 4, none: 5 };
/** Any other field a pane shows is the custom one, drawn from the `custom` attribute (setCustomField). */
const CUSTOM_FIELD = 6;
const INK = '#0d2233';
const CLAY = '#e4e9ed';
const ACCENT = '#0b6fbf';
const FOV = 30;
const ORTHO_DISTANCE = 20;
/** How close `flyTo` comes, as a share of the distance that frames the whole car. */
const CLOSE_UP = 0.4;
/** The camera counts as moving until this long after its last change. */
const SETTLE_MS = 140;
/** Screen motion per frame, in pixels, that counts as the camera moving (for choosing the mesh). */
const MOVING_PX = 0.5;
/** Screen motion per frame, in pixels, below which a frame is not worth drawing. */
const SUBPIXEL = 0.05;
/** A full-mesh frame slower than this while the camera moves hands motion to the working mesh. */
const MOVING_BUDGET_MS = 14;
const ARRIVAL_MS = 850;
/**
 * Bytes per millisecond handed to the GPU while the car uploads: 0.5 GB/s, 8 MB per 60 Hz frame, which the browser's
 * transfer buffer takes without waiting on the GPU process. Measured during a real load: 8 MB a frame never cost
 * more than 1.1 ms, while 17 MB a frame blocked for up to 60 ms whenever the networks kept the GPU process busy.
 * Paced by time, not by frame, so a 120 Hz display hands over no more per second than a 60 Hz one.
 */
const UPLOAD_RATE = 1 << 19;

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
		/** The working mesh the floor is laid under and picking reads: the surface's, once it is on the GPU. */
		this.placed = null;
		/** Starts the field's arrival once the mesh carrying it is on the GPU. */
		this.arrival = null;
		/** While the networks predict: {front}, the x the predicted share's front is easing toward; else null. */
		this.waiting = null;
		/** The selection shown, per working-mesh vertex (setSelection), or null. */
		this.selection = null;
		/** A field the app supplies (setCustomField), and paint on the car (setPaint), per working-mesh vertex, or null. */
		this.custom = null;
		this.paint = null;
		/** Objects the annotations draw with the car (setAnnotations), or null. */
		this.annotations = null;
		/** What the left button does on the car while a tool owns it (setTool), instead of turning the camera. */
		this.tool = null;
		this.lastUpload = 0;
		/** Signalled once the GPU has taken in the last piece uploaded. */
		this.uploadFence = null;
		this.panes = [{ field: 'none', map: null }];
		this.display = { edges: false, ground: true };
		this.section = null;
		this.markers = null;
		this.animations = new Set();
		this.dirty = true;
		this.pending = false;
		this.lastMotion = -Infinity;
		this.lastFrame = 0;
		/** What the last frame drew: its layer and triangle count. */
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
		// three rebuilds its own state when the GPU hands the context back; the car's buffers and the floor shadow are
		// made again.
		this.canvas.addEventListener('webglcontextrestored', () => {
			this.surface?.restage();
			this.placed = null;
			this.uploadFence = null;
			this.disposeStill();
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
			uSelection: { value: 0 },
			uPaint: { value: 0 },
			uPlane: { value: new THREE.Vector4(0, 1, 0, 0) },
			uArrival: { value: new THREE.Vector4() },
			uFade: { value: 1 },
			uWait: { value: new THREE.Vector4() },
			uWaitTime: { value: 0 },
			uAccent: { value: new THREE.Color(ACCENT) },
		};
		this.carMaterial = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: CAR_VERTEX, fragmentShader: CAR_FRAGMENT, side: THREE.DoubleSide });
		this.car = new THREE.Mesh(new THREE.BufferGeometry(), this.carMaterial);
		this.car.frustumCulled = false;
		this.car.visible = false;
		this.waitInfoMaterial = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: CAR_VERTEX, fragmentShader: WAIT_INFO_FRAGMENT, side: THREE.DoubleSide });
		const u = this.uniforms;
		this.waitComposite = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
			uniforms: { uStill: { value: null }, uInfo: { value: null }, uClay: u.uClay, uAccent: u.uAccent, uWait: u.uWait, uWaitTime: u.uWaitTime },
			vertexShader: SCREEN_VERTEX,
			fragmentShader: WAIT_COMPOSITE_FRAGMENT,
			depthTest: false,
			depthWrite: false,
		}));
		this.waitComposite.frustumCulled = false;
		/** While the car waits for its field: the frame drawn in plain clay and what the tint needs per pixel (keepStill). */
		this.still = null;

		this.edgeMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uInk: this.uniforms.uInk,
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
		this.faces = null;
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
		// The waiting car's two passes, into a target of the kind they use: a program is linked per kind of target.
		const info = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
		renderer.setRenderTarget(info);
		this.car.material = this.waitInfoMaterial;
		renderer.render(this.car, this.camera);
		this.car.material = this.carMaterial;
		renderer.setRenderTarget(null);
		renderer.render(this.waitComposite, this.camera);
		info.dispose();
		this.ground.bakeShadow(renderer, speck, [0, 1, 0, 1, 0]);
		[this.car.geometry, this.edges.geometry, this.pickMesh.geometry] = geometries;
		for (const { o, visible, culled } of objects) Object.assign(o, { visible, frustumCulled: culled });
		renderer.clear();
		speck.dispose();
	}

	// ---- the car ------------------------------------------------------------------------------------------------

	/** Show a case's surface, or nothing (the first screen). It is drawn once its working mesh is on the GPU. */
	show(surface) {
		this.surface = surface;
		this.placed = null;
		this.bounds = surface?.working.bounds ?? null;
		this.grid = null;
		this.faces = null;
		this.ground.object.visible = false;
		this.cancelAnimation('arrival');
		this.arrival = null;
		this.uniforms.uArrival.value.w = 0;
		this.uniforms.uFade.value = 1;
		this.endWait();
		// What the app laid on the last car's working mesh belongs to that car.
		this.selection = null;
		this.custom = null;
		this.paint = null;
		this.uniforms.uSelection.value = 0;
		this.uniforms.uPaint.value = 0;
		this.requestRender();
	}

	/** The surface gained a mesh or fields; the frame loop uploads them and re-lays the floor under a new working mesh. */
	surfaceChanged() {
		this.requestRender();
	}

	/**
	 * Move the next pieces of the car to the GPU. Once the working mesh is there, lay the floor under it and pick from
	 * it; once the fields shown are there too, start a waiting arrival.
	 */
	uploadSurface(now) {
		const s = this.surface;
		if (!s) return;
		if (s.uploading && this.uploadTaken()) {
			// A piece the size of the time since the last one, at most a 60 Hz frame's worth.
			const ms = Math.min(1000 / 60, now - this.lastUpload);
			this.lastUpload = now;
			if (s.upload(this.renderer, ms * UPLOAD_RATE)) this.dirty = true;
			const gl = this.renderer.getContext();
			this.uploadFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
		}
		if (s.working.ready && this.placed !== s.working) {
			this.placeSurface();
			this.dirty = true;
		}
		if (this.arrival && this.placed === s.working && this.panes.every((pane) => s.working.has(pane.field))) {
			const start = this.arrival;
			this.arrival = null;
			start();
		}
	}

	/**
	 * Whether the GPU has taken in the last piece. The GPU process also runs the networks, and a piece handed over
	 * while it is behind blocks the page until it catches up; so the next piece waits for it instead.
	 */
	uploadTaken() {
		const fence = this.uploadFence;
		if (!fence) return true;
		const gl = this.renderer.getContext();
		if (gl.getSyncParameter(fence, gl.SYNC_STATUS) !== gl.SIGNALED) return false;
		gl.deleteSync(fence);
		this.uploadFence = null;
		return true;
	}

	/** The working mesh is on the GPU: lay the floor under it and pick from it. Listeners hear 'placed'. */
	placeSurface() {
		const working = this.surface.working;
		this.placed = working;
		this.bounds = working.bounds;
		this.pickMesh.geometry = working.geometry;
		this.grid = null;
		this.faces = null;
		this.ground.object.visible = this.display.ground;
		this.uniforms.uZmin.value = this.bounds[4];
		this.ground.place(this.renderer, working.geometry, this.bounds);
		this.placePlane();
		this.attachAllToWorking();
		this.dispatchEvent(new CustomEvent('placed'));
	}

	/** The vertex grid and face index are built on first use: picking is the only thing that needs them. */
	vertexGrid() {
		this.grid ??= new VertexGrid(this.placed.mesh.positions);
		return this.grid;
	}

	faceIndex() {
		this.faces ??= new FaceIndex(this.placed.mesh);
		return this.faces;
	}

	/**
	 * Which mesh to draw now: the full mesh when it carries every field shown, unless motion must go faster. While the
	 * networks are still predicting the full mesh they share the GPU with drawing, so motion stays on the working mesh
	 * and its cost on the full mesh is timed only once they finish.
	 */
	chooseLayer(moving) {
		const s = this.surface;
		if (!this.placed) return null;
		const full = s.full?.ready ? s.full : null;
		// A new working mesh still on its way: the full mesh stands in, clay until its fields arrive.
		if (!s.working.ready) return full ? 'full' : null;
		// The full mesh shown as the preview is, until the result arrives, the working mesh too; and a selection, paint
		// and a custom field are carried by the working mesh only.
		if (!full || full === s.working || this.selection || this.paint) return 'working';
		if (!this.panes.every((pane) => full.has(pane.field))) return 'working';
		if (this.lod.layer !== full) this.resetLod(full);
		if (moving && (s.fullFraction < 1 || !this.lod.fullWhileMoving)) return 'working';
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
		this.ground.object.visible = this.placed !== null && this.display.ground;
		this.requestRender();
	}

	/** A section plane normal to an axis: {axis: 0|1|2, offset, plane: whether to draw the plane}, or null. */
	setSection(section) {
		const u = this.uniforms;
		this.section = section;
		u.uSection.value = section ? 1 : 0;
		if (section) {
			const n = [0, 0, 0];
			n[section.axis] = 1;
			u.uPlane.value.set(...n, section.offset);
		}
		this.placePlane();
		this.requestRender();
	}

	/**
	 * Show a selection on the car: `selected` is, per working-mesh vertex, the share of the area around it that is
	 * selected (0 to 1). Faces outside fade toward clay and the selection's edge is drawn. null shows none. The working
	 * mesh is drawn while a selection shows, since only it carries the shares. After changing the array in place, call
	 * again with it.
	 */
	setSelection(selected) {
		this.selection = selected;
		this.uniforms.uSelection.value = selected ? 1 : 0;
		this.attachToWorking('selected', selected);
		this.requestRender();
	}

	/**
	 * A field the app supplies, one value per working-mesh vertex, drawn by any pane whose field the viewer does not
	 * know (with that pane's map); null for none. After changing the array in place, call again with it.
	 */
	setCustomField(values) {
		this.custom = values;
		this.attachToWorking('custom', values);
		this.requestRender();
	}

	/**
	 * Paint on the car: per working-mesh vertex an sRGB colour with its strength as alpha (`colour`, RGBA bytes) and
	 * the share of the area around it covered (`cover`, bytes for 0 to 1); null for none. The working mesh is drawn
	 * while paint shows. After changing the arrays in place, call again with them.
	 */
	setPaint(paint) {
		this.paint = paint;
		this.uniforms.uPaint.value = paint ? 1 : 0;
		this.attachToWorking('paint', paint?.colour, 4, true);
		this.attachToWorking('paintCover', paint?.cover, 1, true);
		this.requestRender();
	}

	/** Put a per-vertex array on the working mesh as an attribute, or remove it for null; it reaches the GPU with the next frame. */
	attachToWorking(name, array, size = 1, normalized = false) {
		const geometry = this.surface?.working.geometry;
		if (!geometry) return;
		const attribute = geometry.getAttribute(name);
		if (!array || array.length !== geometry.getAttribute('position')?.count * size) {
			if (attribute) geometry.deleteAttribute(name);
			return;
		}
		if (attribute?.array === array) attribute.needsUpdate = true;
		else geometry.setAttribute(name, new THREE.BufferAttribute(array, size, normalized));
	}

	/** Every per-vertex array the app supplies, on a working mesh just placed. */
	attachAllToWorking() {
		this.attachToWorking('selected', this.selection);
		this.attachToWorking('custom', this.custom);
		this.attachToWorking('paint', this.paint?.colour, 4, true);
		this.attachToWorking('paintCover', this.paint?.cover, 1, true);
	}

	/** Objects the annotations draw with the car in world space, in every pane, or null. The caller owns and disposes them. */
	setAnnotations(object) {
		if (this.annotations) this.scene.remove(this.annotations);
		this.annotations = object;
		if (object) this.scene.add(object);
		this.requestRender();
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
		const sweep = k === null ? -100 : -5 + 14.5 * k;
		if (this.lattice.uniforms.uSweep.value === sweep) return;
		this.lattice.uniforms.uSweep.value = sweep;
		this.requestRender();
	}

	/**
	 * The field arrives on the car as a front carried downstream, broken into streaks like dye in a tunnel. The car
	 * stays clay until the mesh carrying the field is on the GPU, and the front starts then; it ends the wait.
	 */
	arrive() {
		const u = this.uniforms;
		this.cancelAnimation('arrival');
		const done = () => {
			u.uArrival.value.w = 0;
			u.uFade.value = 1;
			this.endWait();
		};
		if (REDUCED_MOTION.matches) {
			u.uFade.value = 0;
			this.arrival = () => this.animate(200, (k) => (u.uFade.value = k), done, 'arrival');
		} else {
			const [xmin, xmax] = this.bounds;
			const length = xmax - xmin;
			const width = 0.22 * length;
			const depth = 0.1 * length;
			u.uArrival.value.set(xmin - depth, width, depth, 1);
			this.arrival = () => this.animate(ARRIVAL_MS, (k) => (u.uArrival.value.x = xmin - depth + (length + width + depth) * (k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2)), done, 'arrival');
		}
		this.requestRender();
	}

	/**
	 * While the networks predict, the clay car shows the work under way: the share predicted so far (0 to 1) as a
	 * faint tint run downstream from the nose, and soft pulses carried downstream. null when nothing is predicted; a
	 * field waiting to arrive ends the wait itself, as its front replaces it.
	 */
	predicting(share) {
		const u = this.uniforms.uWait.value;
		if (share === null) {
			if (this.waiting && !this.arrival && !this.animating('arrival')) {
				this.endWait();
				this.requestRender();
			}
			return;
		}
		if (!this.bounds) return;
		const [xmin, xmax] = this.bounds;
		const length = xmax - xmin;
		// The front starts ahead of the nose and ends behind the tail, past the streaks' lag either way.
		const front = xmin - 0.1 * length + 1.2 * length * share;
		const started = this.waiting !== null;
		if (!started) u.set(front, xmin, length, 1);
		this.waiting = { front };
		// Once under way the frame loop eases the front toward the new share on the still frame; a redraw of the car
		// is needed only to start, or under reduced motion, where the front moves only with the events.
		if (REDUCED_MOTION.matches) u.x = front;
		if (!started || REDUCED_MOTION.matches) this.requestRender();
		else this.schedule();
	}

	/** Nothing waits for a field any more: the tint goes and the still frame is let go. */
	endWait() {
		this.waiting = null;
		this.uniforms.uWait.value.w = 0;
		this.disposeStill();
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

	/**
	 * Fly to a surface point and look at it from outside, along its normal tipped up to stay above the floor, from
	 * CLOSE_UP of the distance that frames the whole car.
	 */
	flyTo(point, normal) {
		const whole = this.framing([normal[0], normal[1], Math.max(normal[2], 0.25)]);
		const target = new THREE.Vector3(...point);
		const offset = whole.position.clone().sub(whole.target).multiplyScalar(CLOSE_UP);
		this.moveTo({ position: target.clone().add(offset), target, zoom: this.camera.isOrthographicCamera ? whole.zoom / CLOSE_UP : 1 }, true, 600);
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

	/**
	 * Fit the drawing buffer to the view and draw at once. Resizing a WebGL canvas clears it, and waiting for the next
	 * animation frame would show that empty canvas for a frame. A view hidden with its page (0 × 0) keeps its buffer.
	 */
	resize() {
		if (!this.host.clientWidth || !this.host.clientHeight) return;
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
		this.disposeStill();
		this.render();
		this.dispatchEvent(new CustomEvent('resize'));
	}

	// ---- picking ----------------------------------------------------------------------------------------------------

	/**
	 * The working-mesh vertex under a client point: {pane, point (where the cursor meets the surface), vertex}, or
	 * null over the background. Rendered into a one-pixel float target and read back without stalling the page.
	 */
	async pick(clientX, clientY) {
		if (!this.placed) return null;
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
		// The nearest working-mesh vertex and the face the point lies on are found only when a reader asks for them.
		let vertex, face;
		const viewer = this;
		return {
			pane: at.index,
			point,
			get vertex() {
				return (vertex ??= viewer.vertexGrid().nearest(point));
			},
			get face() {
				return (face ??= viewer.faceIndex().faceAt(point, this.vertex));
			},
		};
	}

	/**
	 * Give the left button on the car to a tool (painting, drawing a lasso) instead of the camera, or back with null.
	 * The tool gets {clientX, clientY} at `down`, `move` and `up`; the right button still pans and the wheel zooms.
	 */
	setTool(tool) {
		this.tool = tool;
		this.controls.mouseButtons.LEFT = tool ? -1 : THREE.MOUSE.ROTATE;
		this.canvas.style.cursor = tool ? 'crosshair' : '';
	}

	/** The world length one CSS pixel spans at `point` in the view. */
	worldPerPixel(point) {
		const camera = this.camera;
		const { h } = this.paneSize();
		if (camera.isOrthographicCamera) return (camera.top - camera.bottom) / camera.zoom / h;
		const depth = camera.position.distanceTo(new THREE.Vector3(...point));
		return (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * depth) / h;
	}

	/**
	 * Where the car's surface is seen at each pixel of the view: world x, y, z and 1, or 0 where no surface is, as
	 * {width, height, data} in drawing-buffer pixels from the bottom left, with `toPixel(clientX, clientY)`.
	 */
	async readSurface() {
		const renderer = this.renderer;
		const { W, H } = this.paneSize();
		const target = new THREE.WebGLRenderTarget(W, H, { type: THREE.FloatType });
		renderer.setRenderTarget(target);
		renderer.setClearColor(0x000000, 0);
		renderer.clear();
		renderer.render(this.pickScene, this.camera);
		renderer.setRenderTarget(null);
		const data = new Float32Array(W * H * 4);
		await renderer.readRenderTargetPixelsAsync(target, 0, 0, W, H, data);
		target.dispose();
		const rect = this.canvas.getBoundingClientRect();
		const scale = W / rect.width;
		return { width: W, height: H, data, toPixel: (x, y) => [Math.floor((x - rect.left) * scale), H - 1 - Math.floor((y - rect.top) * scale)] };
	}

	bindPointer() {
		const el = this.canvas;
		let down = null;
		let tapTimer = 0;
		let using = false;
		el.addEventListener('pointerdown', (e) => {
			down = { x: e.clientX, y: e.clientY, button: e.button };
			if (!this.tool || e.button !== 0) return;
			using = true;
			el.setPointerCapture(e.pointerId);
			this.tool.down({ clientX: e.clientX, clientY: e.clientY });
		});
		el.addEventListener('pointerup', (e) => {
			if (!using) return;
			using = false;
			this.tool?.up({ clientX: e.clientX, clientY: e.clientY });
		});
		// A single click waits for a possible second one, so a double-click (pivot) never also counts as a click. What it
		// hit is picked at once, with the camera as it was clicked: the view may move during the wait (a view key).
		el.addEventListener('click', (e) => {
			if (this.tool || !down || down.button !== 0 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
			clearTimeout(tapTimer);
			if (e.detail > 1) return;
			const { clientX, clientY } = e;
			const picked = this.pick(clientX, clientY);
			tapTimer = setTimeout(async () => {
				const hit = await picked;
				this.dispatchEvent(new CustomEvent('tap', { detail: { hit, clientX, clientY } }));
			}, 230);
		});
		el.addEventListener('dblclick', async (e) => {
			clearTimeout(tapTimer);
			if (this.tool) return;
			const hit = await this.pick(e.clientX, e.clientY);
			if (hit) this.focus(hit.point);
		});
		el.addEventListener('pointermove', (e) => {
			if (using) return this.tool?.move({ clientX: e.clientX, clientY: e.clientY });
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

	animating(key) {
		return [...this.animations].some((a) => a.key === key);
	}

	/**
	 * Move the waiting car on by `dt` seconds: its front eases toward the share predicted (events come a few times a
	 * second) and the pulses run on. Under reduced motion it stands still between events. Returns whether it moved.
	 */
	stepWait(dt) {
		if (!this.waiting || !this.placed || REDUCED_MOTION.matches) return false;
		const u = this.uniforms.uWait.value;
		u.x += (this.waiting.front - u.x) * (1 - Math.exp(-dt / 0.25));
		this.uniforms.uWaitTime.value += dt;
		return true;
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
		this.uploadSurface(now);
		const animating = this.animations.size > 0;
		for (const a of [...this.animations]) {
			const k = Math.min(1, (now - a.start) / a.ms);
			a.step(k);
			if (k >= 1) {
				this.animations.delete(a);
				a.done?.();
			}
		}
		const cameraAnimating = this.animating('camera');
		const controlled = !cameraAnimating && this.controls.update();
		// Damping glides on for a second in sub-pixel steps; only motion a viewer can see counts as moving. The camera
		// can also move between frames (a wheel step updates it at once), so motion is measured on screen. The glide is
		// ended once it is invisible, or once the controls stop reporting it (under a millimetre a step), since what
		// is left of it would otherwise be applied by the next update.
		const shift = this.screenMotion();
		if (!cameraAnimating && (!controlled || shift <= SUBPIXEL)) this.stopGlide();
		if (cameraAnimating || shift > MOVING_PX) this.lastMotion = now;
		if (shift > SUBPIXEL && this.pointer && this.hoverPicking) this.hover(this.pointer);
		const dt = Math.min(100, now - this.lastFrame) / 1000;
		const latticeMoving = this.lattice.object.visible && this.backdrop === 'moving';
		if (latticeMoving) this.lattice.uniforms.uTime.value += dt;
		const waitMoving = this.stepWait(dt);
		const moving = now - this.lastMotion < SETTLE_MS;
		const layer = this.chooseLayer(moving);
		const redraw = this.dirty || animating || shift > SUBPIXEL || latticeMoving || layer !== this.drawn?.layer;
		// While the car waits for its field it is drawn once, and each frame after only the tint is laid over it.
		const still = waitMoving && this.panes.length === 1 && layer !== null && !this.animating('arrival');
		if (redraw || (waitMoving && !(still && this.still?.layer === layer))) {
			const start = performance.now();
			if (still) this.keepStill(layer);
			else this.render(layer);
			if (layer === 'full' && moving && !this.lod.decided) this.measure(start);
		}
		if (still) this.composeWait();
		this.lastFrame = now;
		this.dirty = false;
		if (this.animations.size > 0 || controlled || moving || latticeMoving || waitMoving || this.surface?.uploading) this.schedule();
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
	 * Draw the waiting car once: the frame in plain clay, kept as a texture, and for each pixel of the car where its
	 * tint is read and how strongly it shows (WAIT_INFO_FRAGMENT). Each frame after only lays the tint over the kept
	 * frame (composeWait), so the animation costs one pass over the screen however many triangles the car has, and
	 * leaves the GPU to the networks it shares it with.
	 */
	keepStill(layer) {
		const renderer = this.renderer;
		const u = this.uniforms;
		const { W, H } = this.paneSize();
		if (this.still && (this.still.width !== W || this.still.height !== H)) this.disposeStill();
		if (!this.still) {
			const frame = new THREE.FramebufferTexture(W, H);
			const info = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
			const composite = this.waitComposite.material.uniforms;
			composite.uStill.value = frame;
			composite.uInfo.value = info.texture;
			this.still = { width: W, height: H, frame, info, key: null, layer: null };
		}
		const strength = u.uWait.value.w;
		u.uWait.value.w = 0;
		this.render(layer);
		u.uWait.value.w = strength;
		renderer.copyFramebufferToTexture(this.still.frame);
		// The car's pixels change only with the view, the mesh and the section; a fading backdrop leaves them be.
		const camera = this.camera;
		const key = [layer, ...camera.matrixWorld.elements, ...camera.projectionMatrix.elements, ...u.uPlane.value.toArray(), u.uSection.value].join();
		if (this.still.key !== key) {
			// Cleared to transparent black, so a pixel off the car reads 0 where the car's read 1.
			renderer.setRenderTarget(this.still.info);
			this.car.material = this.waitInfoMaterial;
			renderer.render(this.car, camera);
			this.car.material = this.carMaterial;
			renderer.setRenderTarget(null);
			this.still.key = key;
		}
		this.still.layer = layer;
	}

	/** Lay the waiting tint, as it is now, over the kept frame. */
	composeWait() {
		this.renderer.setViewport(0, 0, this.host.clientWidth, this.host.clientHeight);
		this.renderer.render(this.waitComposite, this.camera);
	}

	disposeStill() {
		if (!this.still) return;
		this.still.frame.dispose();
		this.still.info.dispose();
		this.still = null;
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
		this.car.visible = layer !== null;
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
			u.uField.value = SHADER_FIELD[pane.field] ?? CUSTOM_FIELD;
			if (pane.map) u.uMap.value.set(pane.map.lo, pane.map.hi, pane.map.bands, rampRow(pane.map.ramp));
			renderer.render(this.scene, camera);
		});
		renderer.autoClear = true;
		renderer.setScissorTest(false);
		this.drawn = { layer: layer ? layerName : null, triangles: layer ? layer.triangles : 0 };
		this.dispatchEvent(new CustomEvent('camera'));
	}

	/** Draw now at full detail and return the canvas; read it in the same task, before the browser clears it. */
	snapshot() {
		this.render(this.chooseLayer(false));
		return this.canvas;
	}

	dispose() {
		this.disposed = true;
		this.disposeStill();
		this.resizeObserver.disconnect();
		this.controls.dispose();
		this.renderer.dispose();
		this.canvas.remove();
	}
}
