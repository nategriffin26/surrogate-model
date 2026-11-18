// Expressions over the working mesh's faces, as `show_field` takes its `expression` and `paint_faces` its `where`: read
// token by token and evaluated for every face at once, never run as code.

/** The per-face arrays an expression may name, made from the working mesh's faces on first use. */
const VARIABLES = {
	cp: (cells) => cells.cp,
	sigma: (cells) => cells.sigma,
	cfd: (cells) => {
		if (!cells.cfd)
			throw new Error('This case has no reference field, so cfd is not defined. Attach one, or leave cfd out.');
		return cells.cfd;
	},
	drag: (cells) => cells.cp.map((cp, f) => -cp * cells.normals[3 * f]),
	x: (cells) => component(cells.centers, 0),
	y: (cells) => component(cells.centers, 1),
	z: (cells) => component(cells.centers, 2),
	nx: (cells) => component(cells.normals, 0),
	ny: (cells) => component(cells.normals, 1),
	nz: (cells) => component(cells.normals, 2),
	area: (cells) => Float32Array.from(cells.areas),
};
const component = (array, k) => Float32Array.from({ length: array.length / 3 }, (_, f) => array[3 * f + k]);

const FUNCTIONS = {
	abs: [1, Math.abs],
	sqrt: [1, Math.sqrt],
	exp: [1, Math.exp],
	log: [1, Math.log],
	log10: [1, Math.log10],
	sign: [1, Math.sign],
	min: [2, Math.min],
	max: [2, Math.max],
	clip: [3, (v, lo, hi) => Math.min(hi, Math.max(lo, v))],
};
const COMPARE = {
	'<': (a, b) => +(a < b),
	'<=': (a, b) => +(a <= b),
	'>': (a, b) => +(a > b),
	'>=': (a, b) => +(a >= b),
	'==': (a, b) => +(a === b),
	'!=': (a, b) => +(a !== b),
};
const ARITHMETIC = {
	'+': (a, b) => a + b,
	'-': (a, b) => a - b,
	'*': (a, b) => a * b,
	'/': (a, b) => a / b,
	'^': (a, b) => a ** b,
};

/**
 * An expression over the faces, evaluated for each: numbers, the names in VARIABLES, + − * / ^ (or **), comparisons
 * (true is 1), and, or, not, parentheses and the functions in FUNCTIONS. Parsed here, never run as code.
 */
export function evaluate(text, cells) {
	const tokens = [];
	const pattern =
		/\s*(?:(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+(?:e[+-]?\d+)?)|([A-Za-z_]\w*)|(\*\*|<=|>=|==|!=|[-+*/^()<>,]))/iy;
	let at = 0;
	const source = String(text);
	while (at < source.length) {
		if (/^\s*$/.test(source.slice(at))) break;
		pattern.lastIndex = at;
		const m = pattern.exec(source);
		if (!m) throw new Error(`The expression cannot be read at "${source.slice(at).trim().slice(0, 20)}".`);
		tokens.push(
			m[1] !== undefined
				? { number: Number(m[1]) }
				: m[2] !== undefined
					? { name: m[2] }
					: { op: m[3] === '**' ? '^' : m[3] },
		);
		at = pattern.lastIndex;
	}
	const faces = cells.areas.length;
	const known = new Map();
	let i = 0;
	const peek = () => tokens[i];
	const isOp = (op) => tokens[i]?.op === op;
	const isWord = (word) => tokens[i]?.name === word;
	const expect = (op) => {
		if (!isOp(op))
			throw new Error(
				`The expression needs "${op}" ${tokens[i] ? 'before' : 'at the end'}${tokens[i] ? ` "${tokens[i].op ?? tokens[i].name ?? tokens[i].number}"` : ''}.`,
			);
		i++;
	};
	// Each node is a number or a Float32Array with one value per face.
	const map = (f, ...args) => {
		if (args.every((a) => typeof a === 'number')) return f(...args);
		const out = new Float32Array(faces);
		for (let k = 0; k < faces; k++) out[k] = f(...args.map((a) => (typeof a === 'number' ? a : a[k])));
		return out;
	};
	const either = () => {
		let left = both();
		while (isWord('or')) (i++, (left = map((a, b) => +(a !== 0 || b !== 0), left, both())));
		return left;
	};
	const both = () => {
		let left = negation();
		while (isWord('and')) (i++, (left = map((a, b) => +(a !== 0 && b !== 0), left, negation())));
		return left;
	};
	const negation = () => (isWord('not') ? (i++, map((a) => +(a === 0), negation())) : comparison());
	const comparison = () => {
		const left = sum();
		const op = peek()?.op;
		if (!COMPARE[op]) return left;
		i++;
		return map(COMPARE[op], left, sum());
	};
	const sum = () => {
		let left = product();
		while (isOp('+') || isOp('-')) left = map(ARITHMETIC[tokens[i++].op], left, product());
		return left;
	};
	const product = () => {
		let left = unary();
		while (isOp('*') || isOp('/')) left = map(ARITHMETIC[tokens[i++].op], left, unary());
		return left;
	};
	const unary = () => (isOp('-') ? (i++, map((a) => -a, unary())) : isOp('+') ? (i++, unary()) : power());
	const power = () => {
		const base = atom();
		return isOp('^') ? (i++, map(ARITHMETIC['^'], base, unary())) : base;
	};
	const atom = () => {
		const token = tokens[i++];
		if (!token) throw new Error('The expression ends too soon.');
		if (token.number !== undefined) return token.number;
		if (token.op === '(') {
			const inside = either();
			expect(')');
			return inside;
		}
		if (token.name !== undefined) {
			const name = token.name.toLowerCase();
			if (FUNCTIONS[name]) {
				const [arity, f] = FUNCTIONS[name];
				expect('(');
				const args = [either()];
				while (isOp(',')) (i++, args.push(either()));
				expect(')');
				if (args.length !== arity)
					throw new Error(`${name} takes ${arity} argument${arity > 1 ? 's' : ''}; got ${args.length}.`);
				return map(f, ...args);
			}
			if (VARIABLES[name]) {
				if (!known.has(name)) known.set(name, VARIABLES[name](cells));
				return known.get(name);
			}
			throw new Error(
				`The expression names "${token.name}", which is not known. Names: ${Object.keys(VARIABLES).join(', ')}; functions: ${Object.keys(FUNCTIONS).join(', ')}.`,
			);
		}
		throw new Error(`The expression has "${token.op}" where a value should be.`);
	};
	const value = either();
	if (i < tokens.length)
		throw new Error(
			`The expression has more after its end, from "${tokens[i].op ?? tokens[i].name ?? tokens[i].number}".`,
		);
	return typeof value === 'number' ? new Float32Array(faces).fill(value) : value;
}
