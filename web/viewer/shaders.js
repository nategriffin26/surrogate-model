// GLSL for the car, its mesh edges, picking and the ground. Fields are per-vertex attributes; every colour
// mapping, the section line, clipping and the arrival of a field are uniforms, so none of them touches a buffer.

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
varying float vCp;
varying float vSigma;
varying float vCfd;
varying vec3 vWorld;
varying vec3 vNormal;
void main() {
	vCp = cp;
	vSigma = sigma;
	vCfd = cfd;
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	vNormal = normal;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

export const CAR_FRAGMENT = /* glsl */ `
uniform sampler2D uRamp;
uniform int uField;          // 0 cp, 1 cfd, 2 cp - cfd, 3 sigma, 4 -cp nx, 5 none
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
uniform float uSection;      // 1 draws the section line
uniform float uEmphasis;     // faces to pick out by the sign of their outward n_x: -1 forward-facing, 1 rearward-facing, 0 none
uniform vec4 uArrival;       // x of the front, front width, streak depth, 1 while the field is arriving
uniform float uFade;         // 0..1 fade from clay, used instead of the front under reduced motion
${CLIP}
varying float vCp;
varying float vSigma;
varying float vCfd;
varying vec3 vWorld;
varying vec3 vNormal;

float fieldValue() {
	if (uField == 0) return vCp;
	if (uField == 1) return vCfd;
	if (uField == 2) return vCp - vCfd;
	if (uField == 3) return vSigma;
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

void main() {
	if (clipped(vWorld)) discard;
	vec3 base = uField == 5 ? uClay : fieldColour();
	if (uArrival.w > 0.0) {
		float lag = uArrival.z * (streaks(vWorld.yz * 9.0) * 0.65 + streaks(vWorld.yz * 23.0) * 0.35);
		float behind = 1.0 - smoothstep(-uArrival.y, 0.0, vWorld.x + lag - uArrival.x);
		float crest = exp(-pow((vWorld.x + lag - uArrival.x + 0.35 * uArrival.y) / (0.35 * uArrival.y), 2.0));
		base = mix(uClay, base, behind) + 0.12 * crest * (1.0 - behind * 0.5);
	}
	if (uFade < 1.0) base = mix(uClay, base, uFade);
	if (uEmphasis != 0.0 && vNormal.x * uEmphasis <= 0.0) base = mix(base, uClay, 0.85);

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
	if (uSection > 0.5) colour = mix(colour, uInk, 0.9 * lineAt(dot(vWorld, uPlane.xyz) - uPlane.w, 2.0));
	gl_FragColor = vec4(colour, 1.0);
	#include <colorspace_fragment>
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
