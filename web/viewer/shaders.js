// GLSL for the car, its mesh edges, picking and the ground. Fields are per-vertex attributes; every colour
// mapping, the section lines, clipping and the arrival of a field are uniforms, so none of them touches a buffer.

/** Sections kept in the plot beside the live one, each drawn on the car in its plot colour. */
export const MAX_KEPT = 6;

/** Shared by the car and the picking pass: clip against the section plane. */
const CLIP = /* glsl */ `
uniform vec4 uPlane;     // section plane: unit normal, offset
uniform float uClip;     // 0 off, 1 keep the side below the offset, -1 keep the side above
bool clipped(vec3 world) {
	return uClip != 0.0 && uClip * (dot(world, uPlane.xyz) - uPlane.w) > 0.0;
}`;

export const CAR_VERTEX = /* glsl */ `
attribute float cp;
attribute float sigma;
attribute float cfd;
attribute float selected;    // share of the area around the vertex inside the selection (working mesh only)
attribute float custom;      // a field the app supplies, such as an annotation's array (working mesh only)
attribute vec4 paint;        // a colour painted on the car, alpha its strength (working mesh only)
attribute float paintCover;  // share of the area around the vertex the paint covers (working mesh only)
varying float vCp;
varying float vSigma;
varying float vCfd;
varying float vSelected;
varying float vCustom;
varying vec4 vPaint;
varying float vPaintCover;
varying vec3 vWorld;
varying vec3 vNormal;
void main() {
	vCp = cp;
	vSigma = sigma;
	vCfd = cfd;
	vSelected = selected;
	vCustom = custom;
	vPaint = paint;
	vPaintCover = paintCover;
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	vNormal = normal;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

/**
 * The clay while the networks predict, at a point whose x has been moved by the streaks: the share predicted so far
 * as a tint run downstream from the nose, and soft pulses carried downstream in the same streaks the field will
 * arrive in, like dye let into the tunnel. Needs uClay, uAccent, uWait and uWaitTime.
 */
const WAITING_CLAY = /* glsl */ `
vec3 waitingClay(float x) {
	float done = 1.0 - smoothstep(-0.05 * uWait.z, 0.0, x - uWait.x);
	float phase = fract((x - uWait.y) / (0.5 * uWait.z) - 0.6 * uWaitTime);
	float pulse = exp(-pow((phase - 0.5) / 0.06, 2.0));
	return mix(uClay, uAccent, 0.2 * done + 0.28 * pulse) + 0.06 * pulse;
}`;

/** Uniforms, inputs and functions of the car's fragment shaders: the colour of a field, and its shading. */
const CAR_COMMON = /* glsl */ `
uniform sampler2D uRamp;
uniform int uField;          // 0 cp, 1 cfd, 2 cp - cfd, 3 sigma, 4 -cp nx, 5 none, 6 the custom field
uniform vec4 uMap;           // lo, hi, bands (0 = continuous), ramp row
uniform vec3 uClay;
uniform vec3 uInk;
uniform vec3 uKey;
uniform vec3 uFill;
uniform float uShade;
uniform float uGloss;
uniform float uZmin;
uniform float uOrtho;
uniform vec3 uViewDir;
uniform float uSection;      // 1 draws the section line, and the kept sections' lines
#define MAX_KEPT ${MAX_KEPT}
uniform vec4 uKept[MAX_KEPT];        // kept sections: unit normal, offset
uniform vec3 uKeptColour[MAX_KEPT];
uniform int uKeptCount;
uniform float uSelection;    // 1 shows the selection: faces outside it fade toward clay, its edge drawn in the accent
uniform float uPaint;        // 1 shows paint on the car
uniform vec4 uArrival;       // x of the front, front width, streak depth, 1 while the field is arriving
uniform float uFade;         // 0..1 fade from clay, used instead of the front under reduced motion
uniform vec4 uWait;          // while the networks predict: x of the predicted share's front, x of the nose, car length, 1
uniform float uWaitTime;     // seconds, carrying the pulses downstream; still under reduced motion
uniform vec3 uAccent;
${CLIP}
varying float vCp;
varying float vSigma;
varying float vCfd;
varying float vSelected;
varying float vCustom;
varying vec4 vPaint;
varying float vPaintCover;
varying vec3 vWorld;
varying vec3 vNormal;

float fieldValue() {
	if (uField == 0) return vCp;
	if (uField == 1) return vCfd;
	if (uField == 2) return vCp - vCfd;
	if (uField == 3) return vSigma;
	if (uField == 6) return vCustom;
	return -vCp * vNormal.x;
}

vec3 fieldColour() {
	float t = clamp((fieldValue() - uMap.x) / (uMap.y - uMap.x), 0.0, 1.0);
	if (uMap.z > 0.0) t = (min(uMap.z - 1.0, floor(t * uMap.z)) + 0.5) / uMap.z;
	return texture2D(uRamp, vec2(0.5 / 256.0 + t * 255.0 / 256.0, uMap.w)).rgb;
}

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Smooth noise across the flow (y, z) only, so the arriving front breaks into streaks that run downstream.
float streaks(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

float lineAt(float s, float px) {
	float w = fwidth(s);
	return 1.0 - smoothstep(w * px * 0.5, w * (px * 0.5 + 1.0), abs(s));
}

${WAITING_CLAY}

/** Smooth noise across the flow, by which a front or a pulse lags as it is carried downstream. */
float streakNoise() {
	return streaks(vWorld.yz * 9.0) * 0.65 + streaks(vWorld.yz * 23.0) * 0.35;
}

/** A base colour lit and reflected, darker inside the shell, under the section line. Affine in the base colour. */
vec3 shaded(vec3 base) {
	vec3 N = normalize(vNormal);
	if (!gl_FrontFacing) N = -N;
	vec3 V = uOrtho > 0.5 ? -uViewDir : normalize(cameraPosition - vWorld);
	float hemi = 0.5 + 0.5 * N.z;
	float key = max(dot(N, uKey), 0.0);
	float fill = max(dot(N, uFill), 0.0);
	float nearGround = mix(0.72, 1.0, smoothstep(0.0, 0.25, vWorld.z - uZmin));
	float light = (0.4 + 0.3 * hemi + 0.48 * key + 0.14 * fill) * mix(1.0, nearGround, step(N.z, 0.25));
	vec3 colour = base * mix(1.0, light, uShade);
	// Studio reflection: a bright soft sky above the horizon, the pale floor below, strongest at grazing angles.
	vec3 R = reflect(-V, N);
	float sky = smoothstep(-0.08, 0.22, R.z);
	float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
	colour = mix(colour, mix(colour * 0.8, vec3(1.0), sky), uGloss * fresnel * 0.9);
	vec3 H = normalize(uKey + V);
	colour += uGloss * 0.35 * pow(max(dot(N, H), 0.0), 80.0);
	// The inside of the shell, seen through a clip, reads darker than the outside.
	if (!gl_FrontFacing) colour *= 0.62;
	if (uSection > 0.5) {
		colour = mix(colour, uInk, 0.9 * lineAt(dot(vWorld, uPlane.xyz) - uPlane.w, 2.0));
		for (int i = 0; i < MAX_KEPT; i++) {
			if (i >= uKeptCount) break;
			colour = mix(colour, uKeptColour[i], 0.9 * lineAt(dot(vWorld, uKept[i].xyz) - uKept[i].w, 2.0));
		}
	}
	return colour;
}`;

export const CAR_FRAGMENT = /* glsl */ `${CAR_COMMON}
void main() {
	if (clipped(vWorld)) discard;
	float noise = uArrival.w > 0.0 || uWait.w > 0.0 ? streakNoise() : 0.0;
	vec3 clay = uWait.w > 0.0 ? waitingClay(vWorld.x + 0.1 * uWait.z * noise) : uClay;
	vec3 base = uField == 5 ? clay : fieldColour();
	if (uArrival.w > 0.0) {
		float lag = uArrival.z * noise;
		float behind = 1.0 - smoothstep(-uArrival.y, 0.0, vWorld.x + lag - uArrival.x);
		float crest = exp(-pow((vWorld.x + lag - uArrival.x + 0.35 * uArrival.y) / (0.35 * uArrival.y), 2.0));
		base = mix(clay, base, behind) + 0.12 * crest * (1.0 - behind * 0.5);
	}
	if (uFade < 1.0) base = mix(clay, base, uFade);
	// Paint arrives as sRGB bytes, like the theme's colours; the car is shaded in linear light.
	if (uPaint > 0.5) base = mix(base, sRGBTransferEOTF(vec4(vPaint.rgb, 1.0)).rgb, vPaint.a * smoothstep(0.3, 0.7, vPaintCover));
	if (uSelection > 0.5) {
		base = mix(mix(base, uClay, 0.8), base, smoothstep(0.3, 0.7, vSelected));
		base = mix(base, uAccent, 0.9 * lineAt(vSelected - 0.5, 1.5));
	}
	gl_FragColor = vec4(shaded(base), 1.0);
	#include <colorspace_fragment>
}`;

/**
 * The waiting car's still, per pixel of the car: the x its tint is read at (streaks included), and how much the
 * colour shown moves with the base colour. Shading is affine in the base, so the composite below can add the
 * waiting tint to a frame drawn once in plain clay and get what the car shader would draw.
 */
export const WAIT_INFO_FRAGMENT = /* glsl */ `${CAR_COMMON}
void main() {
	if (clipped(vWorld)) discard;
	float gain = shaded(vec3(1.0)).g - shaded(vec3(0.0)).g;
	gl_FragColor = vec4(vWorld.x + 0.1 * uWait.z * streakNoise(), gain, 1.0, 1.0);
}`;

export const SCREEN_VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

/** A waiting frame: the still frame, as the canvas held it, with the waiting tint added over the car's pixels. */
export const WAIT_COMPOSITE_FRAGMENT = /* glsl */ `
uniform sampler2D uStill;
uniform sampler2D uInfo;
uniform vec3 uClay;
uniform vec3 uAccent;
uniform vec4 uWait;
uniform float uWaitTime;
varying vec2 vUv;
${WAITING_CLAY}
vec3 decode(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec3 encode(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
void main() {
	vec4 still = texture2D(uStill, vUv);
	vec4 info = texture2D(uInfo, vUv);
	if (info.z > 0.5) still.rgb = encode(max(decode(still.rgb) + (waitingClay(info.x) - uClay) * info.y, 0.0));
	gl_FragColor = still;
}`;

export const EDGE_VERTEX = /* glsl */ `
attribute float edge;        // mean length of the edges meeting at this vertex, metres
uniform float uPixel;        // world size of one pixel at unit distance (perspective) or anywhere (orthographic)
uniform float uOrtho;
varying float vPixels;
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	vec4 eye = viewMatrix * world;
	vPixels = edge / (uOrtho > 0.5 ? uPixel : uPixel * -eye.z);
	gl_Position = projectionMatrix * eye;
}`;

export const EDGE_FRAGMENT = /* glsl */ `
uniform vec3 uInk;
${CLIP}
varying float vPixels;
varying vec3 vWorld;
void main() {
	if (clipped(vWorld)) discard;
	// Edges fade out where they would crowd closer than a few pixels and turn the surface solid.
	float a = smoothstep(3.0, 9.0, vPixels) * 0.5;
	if (a <= 0.0) discard;
	gl_FragColor = vec4(uInk, a);
	#include <colorspace_fragment>
}`;

export const PICK_VERTEX = /* glsl */ `
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

export const PICK_FRAGMENT = /* glsl */ `
${CLIP}
varying vec3 vWorld;
void main() {
	if (clipped(vWorld)) discard;
	gl_FragColor = vec4(vWorld, 1.0);
}`;

export const GROUND_VERTEX = /* glsl */ `
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

export const GROUND_FRAGMENT = /* glsl */ `
uniform sampler2D uShadow;
uniform sampler2D uShadowWide;
uniform mat4 uShadowMatrix;
uniform float uShadowAmount;
uniform float uGrid;
uniform vec3 uGridColour;
uniform vec3 uShadowColour;
uniform vec2 uCentre;
uniform vec2 uHalf;
uniform float uOrtho;
uniform vec3 uViewDir;
varying vec3 vWorld;
float family(vec2 p, vec2 n, float s) {
	float u = dot(p, n) / s;
	float w = fwidth(u);
	return (1.0 - smoothstep(0.0, w * 1.2, abs(fract(u - 0.5) - 0.5))) * (1.0 - smoothstep(0.12, 0.4, w));
}
void main() {
	vec4 sp = uShadowMatrix * vec4(vWorld, 1.0);
	vec2 uv = sp.xy * 0.5 + 0.5;
	float inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
	float sa = clamp(0.75 * texture2D(uShadow, uv).r + 0.5 * texture2D(uShadowWide, uv).r, 0.0, 1.0) * inside * uShadowAmount;
	vec2 p = vWorld.xy;
	float tri = max(max(family(p, vec2(0.0, 1.0), 0.3), family(p, vec2(0.8660254, 0.5), 0.3)), family(p, vec2(0.8660254, -0.5), 0.3));
	float fade = 1.0 - smoothstep(0.55, 1.0, length((p - uCentre) / uHalf));
	// Seen edge-on the lattice turns to noise, so it fades as the view flattens.
	vec3 toEye = uOrtho > 0.5 ? -uViewDir : normalize(cameraPosition - vWorld);
	fade *= smoothstep(0.12, 0.35, abs(toEye.z));
	float ga = tri * uGrid * fade * 0.42;
	float a = sa + ga * (1.0 - sa);
	vec3 c = (uShadowColour * sa + uGridColour * ga * (1.0 - sa)) / max(a, 1e-4);
	gl_FragColor = vec4(c, a);
	#include <colorspace_fragment>
}`;

export const LATTICE_VERTEX = /* glsl */ `
uniform float uTime;
uniform float uAmp;
uniform float uSweep;
uniform vec2 uCentre;
varying float vFade;
varying float vBand;
void main() {
	vec3 p = position;
	float wave = sin(p.x * 0.9 + uTime * 0.55) * cos(p.y * 1.25 - uTime * 0.4) + 0.6 * sin(p.x * 0.33 - p.y * 0.7 + uTime * 0.3);
	p.z += uAmp * wave * smoothstep(0.0, 2.5, length(p.xy - uCentre) - 1.2);
	vFade = 1.0 - smoothstep(1.5, 5.2, length((p.xy - uCentre) * vec2(0.72, 1.25)));
	vBand = exp(-pow((p.x - uSweep) / 0.45, 2.0));
	gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;

export const LATTICE_FRAGMENT = /* glsl */ `
uniform vec3 uColour;
uniform vec3 uBandColour;
uniform float uOpacity;
varying float vFade;
varying float vBand;
void main() {
	gl_FragColor = vec4(mix(uColour, uBandColour, vBand), (0.2 + 0.55 * vBand) * vFade * uOpacity);
	#include <colorspace_fragment>
}`;

export const SHADOW_DEPTH_VERTEX = /* glsl */ `
uniform float uZmin;
varying float vHeight;
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	vHeight = world.z - uZmin;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

export const SHADOW_DEPTH_FRAGMENT = /* glsl */ `
uniform float uHeight;
${CLIP}
varying float vHeight;
varying vec3 vWorld;
void main() {
	if (clipped(vWorld)) discard;
	float d = 1.0 - clamp(vHeight / uHeight, 0.0, 1.0);
	gl_FragColor = vec4(d * d, 0.0, 0.0, 1.0);
}`;

export const BLUR_FRAGMENT = /* glsl */ `
uniform sampler2D uSource;
uniform vec2 uStep;
varying vec2 vUv;
void main() {
	float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
	float s = texture2D(uSource, vUv).r * w[0];
	for (int i = 1; i < 5; i++) {
		s += texture2D(uSource, vUv + uStep * float(i)).r * w[i];
		s += texture2D(uSource, vUv - uStep * float(i)).r * w[i];
	}
	gl_FragColor = vec4(s, 0.0, 0.0, 1.0);
}`;
