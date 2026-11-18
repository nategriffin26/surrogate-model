// Number formats for everything on screen. Negative numbers use the minus sign; files use plain ASCII.

const MINUS = '\u2212';

/** A number with a typographic minus; one that rounds to zero has none. */
export const signed = (x, digits = 3) => {
	const magnitude = Math.abs(x).toFixed(digits);
	return `${x < 0 && Number(magnitude) !== 0 ? MINUS : ''}${magnitude}`;
};
export const seconds = (s) => (s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`);
export const count = (n) => n.toLocaleString('en-US');
export const millions = (n) => `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)} M`;

export function bytes(n) {
	if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
	if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
	if (n >= 1e3) return `${Math.round(n / 1e3)} kB`;
	return `${n} B`;
}

/** Digits that resolve a step on an axis or a legend. */
export function digitsFor(step) {
	return Math.max(0, Math.min(4, Math.ceil(-Math.log10(step) - 1e-9)));
}

/**
 * Round, readable tick values covering [lo, hi], about `target` of them, and the decimals that write each one
 * exactly: a step of 0.25 needs two, where `digitsFor` would give one and print 0.25 as 0.3.
 */
export function niceTicks(lo, hi, target = 6) {
	const span = hi - lo;
	if (!(span > 0)) return { step: 1, ticks: [lo], digits: digitsFor(Math.abs(lo) || 1) };
	const raw = span / target;
	const exponent = Math.floor(Math.log10(raw));
	const power = 10 ** exponent;
	const multiple = [1, 2, 2.5, 5, 10].find((m) => m * power >= raw);
	const step = multiple * power;
	const ticks = [];
	for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
	return { step, ticks, digits: Math.max(0, -exponent - (multiple === 10 ? 1 : 0) + (multiple === 2.5 ? 1 : 0)) };
}
