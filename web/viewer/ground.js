// The studio floor under the car (a contact shadow and a faint triangular grid) and the moving lattice behind the
// first screen and the wait for a file.

import * as THREE from 'three';
import {
	BLUR_FRAGMENT, GROUND_FRAGMENT, GROUND_VERTEX, LATTICE_FRAGMENT, LATTICE_VERTEX, SHADOW_DEPTH_FRAGMENT, SHADOW_DEPTH_VERTEX,
} from './shaders.js';

export class Ground {
	constructor(shared) {
		this.uniforms = {
			uShadow: { value: null },
			uShadowWide: { value: null },
			uShadowMatrix: { value: new THREE.Matrix4() },
			uShadowAmount: { value: 0.7 },
			uGrid: { value: 0.25 },
			uGridColour: { value: new THREE.Color('#2f739c') },
			uShadowColour: { value: new THREE.Color('#0d2233') },
			uCentre: { value: new THREE.Vector2(1.6, 0) },
			uHalf: { value: new THREE.Vector2(6, 4) },
			uOrtho: shared.uOrtho,
			uViewDir: shared.uViewDir,
		};
		this.object = new THREE.Mesh(
			new THREE.PlaneGeometry(60, 60),
			new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: GROUND_VERTEX, fragmentShader: GROUND_FRAGMENT, transparent: true, depthWrite: false }),
		);
		this.object.renderOrder = 1;
		this.object.visible = false;
		this.clip = { uPlane: shared.uPlane, uClip: shared.uClip };

		// The shadow bake's scenes and materials live as long as the floor: making them per bake would compile their
		// shader programs again on every case switch and every clip change.
		this.depth = new THREE.ShaderMaterial({
			uniforms: { uZmin: { value: 0 }, uHeight: { value: 0.7 }, ...this.clip },
			vertexShader: SHADOW_DEPTH_VERTEX,
			fragmentShader: SHADOW_DEPTH_FRAGMENT,
			side: THREE.DoubleSide,
		});
		this.emptyGeometry = new THREE.BufferGeometry();
		this.depthCar = new THREE.Mesh(this.emptyGeometry, this.depth);
		this.depthCar.frustumCulled = false;
		this.depthScene = new THREE.Scene().add(this.depthCar);
		this.depthCamera = new THREE.OrthographicCamera();
		this.blur = new THREE.ShaderMaterial({
			uniforms: { uSource: { value: null }, uStep: { value: new THREE.Vector2() } },
			vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
			fragmentShader: BLUR_FRAGMENT,
		});
		this.blurScene = new THREE.Scene().add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.blur));
		this.blurCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
		this.targets = [];
	}

	/** Lay the floor under a car and bake its contact shadow from the car's geometry, as clipped by the section. */
	place(renderer, geometry, bounds) {
		const [xmin, xmax, ymin, ymax, zmin] = bounds;
		this.object.position.z = zmin;
		this.uniforms.uCentre.value.set((xmin + xmax) / 2, 0);
		this.uniforms.uHalf.value.set((xmax - xmin) * 1.05 + 1.5, (ymax - ymin) * 1.2 + 1.5);
		this.bakeShadow(renderer, geometry, bounds);
	}

	bakeShadow(renderer, geometry, [xmin, xmax, ymin, ymax, zmin]) {
		const pad = 0.7;
		const w = xmax - xmin + 2 * pad;
		const h = ymax - ymin + 2 * pad;
		const W = 1024;
		const H = Math.round((W * h) / w);
		const camera = this.depthCamera;
		Object.assign(camera, { left: -w / 2, right: w / 2, top: h / 2, bottom: -h / 2, near: 0, far: 0.75 });
		camera.updateProjectionMatrix();
		camera.position.set((xmin + xmax) / 2, (ymin + ymax) / 2, zmin - 0.0005);
		camera.up.set(0, 1, 0);
		camera.lookAt(camera.position.x, camera.position.y, zmin + 1);
		camera.updateMatrixWorld();
		this.depth.uniforms.uZmin.value = zmin;
		this.depthCar.geometry = geometry;
		if (this.targets[0]?.height !== H) {
			for (const target of this.targets) target.dispose();
			const options = { type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
			this.targets = [0, 1, 2].map(() => new THREE.WebGLRenderTarget(W, H, options));
		}
		const [A, B, C] = this.targets;
		renderer.setRenderTarget(A);
		renderer.setClearColor(0x000000, 1);
		renderer.clear();
		renderer.render(this.depthScene, camera);

		const pass = (source, target, radius, horizontal) => {
			this.blur.uniforms.uSource.value = source.texture;
			this.blur.uniforms.uStep.value.set(horizontal ? radius / W : 0, horizontal ? 0 : radius / H);
			renderer.setRenderTarget(target);
			renderer.render(this.blurScene, this.blurCamera);
		};
		// A: a tight contact shadow under the tyres and sills. C: a wide soft shadow under the whole body.
		for (const radius of [2, 4]) {
			pass(A, B, radius, true);
			pass(B, A, radius, false);
		}
		pass(A, B, 8, true);
		pass(B, C, 8, false);
		for (const radius of [16, 26]) {
			pass(C, B, radius, true);
			pass(B, C, radius, false);
		}
		renderer.setRenderTarget(null);
		renderer.setClearColor(0x000000, 0);
		// The bake holds no geometry between bakes, so a closed case's arrays are not kept alive by it.
		this.depthCar.geometry = this.emptyGeometry;
		this.uniforms.uShadow.value = A.texture;
		this.uniforms.uShadowWide.value = C.texture;
		this.uniforms.uShadowMatrix.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
	}
}

/** A triangular lattice, Rescale's wireframe-mesh motif, laid on the floor and gently rolling. */
export class Lattice {
	constructor() {
		const step = 0.32;
		const cols = 46;
		const rows = 30;
		const points = [];
		const at = (i, j) => j * cols + i;
		for (let j = 0; j < rows; j++) {
			for (let i = 0; i < cols; i++) points.push(-5.6 + i * step + (j % 2) * step * 0.5, (j - (rows - 1) / 2) * step * 0.866, 0);
		}
		const lines = [];
		for (let j = 0; j < rows; j++) {
			for (let i = 0; i < cols; i++) {
				if (i + 1 < cols) lines.push(at(i, j), at(i + 1, j));
				if (j + 1 < rows) {
					lines.push(at(i, j), at(i, j + 1));
					const k = j % 2 === 0 ? i - 1 : i + 1;
					if (k >= 0 && k < cols) lines.push(at(i, j), at(k, j + 1));
				}
			}
		}
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
		geometry.setIndex(lines);
		this.uniforms = {
			uTime: { value: 0 },
			uAmp: { value: 0.18 },
			uSweep: { value: -100 },
			uCentre: { value: new THREE.Vector2(1.6, 0) },
			uColour: { value: new THREE.Color('#2f739c') },
			uBandColour: { value: new THREE.Color('#0b6fbf') },
			uOpacity: { value: 1 },
		};
		this.object = new THREE.LineSegments(
			geometry,
			new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: LATTICE_VERTEX, fragmentShader: LATTICE_FRAGMENT, transparent: true, depthWrite: false }),
		);
		this.object.position.z = -0.3175;
		this.object.renderOrder = 2;
	}
}
