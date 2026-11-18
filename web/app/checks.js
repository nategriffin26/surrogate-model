// Checks on an operation's arguments, whose errors say what was wrong and what would work. The assistant reads
// these messages and corrects its call, so each names the argument, what it got and what it accepts.

export function oneOf(name, value, options) {
	if (!options.includes(value)) throw new Error(`${name} must be one of ${options.map((o) => JSON.stringify(o)).join(', ')}; got ${JSON.stringify(value)}.`);
	return value;
}

export function finite(name, value) {
	if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${name} must be a number; got ${JSON.stringify(value)}.`);
	return value;
}

/** A point or direction [x, y, z] in metres along the file's axes. */
export function triple(name, value) {
	if (!Array.isArray(value) || value.length !== 3 || !value.every((v) => typeof v === 'number' && Number.isFinite(v))) {
		throw new Error(`${name} must be three numbers [x, y, z]; got ${JSON.stringify(value)}.`);
	}
	return value;
}

/** Round for a result the assistant reads: enough digits for any value on screen, none of float noise. */
export const round = (value, digits = 6) => Number(value.toFixed(digits));
