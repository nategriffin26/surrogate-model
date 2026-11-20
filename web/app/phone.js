// The tool on a phone. The car fills the screen under the top bar, and one sheet over its foot holds everything else,
// as a map app keeps its map whole under one sheet: the field switch and the whole car's headline numbers at its low
// detent; raised, tabs in the headline's place for the numbers, the section, the probes and the variants; and the
// Files, Assistant, Model and Export pages, opened from the top bar's menu, in the same sheet. The legend and the view
// tools float over the car.
//
// The sheet holds the tool's own elements, moved into it while the window is a phone's and put back where they came
// from when it is not, so a desktop window lays out exactly as it would without this module. Held sideways, the sheet
// is a column beside the car.

import { coefficient, h, segmented } from './dom.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';

/** A phone: a narrow window, or a touch screen held sideways. app.css lays the phone out under the same queries. */
const PHONE = matchMedia('(max-width: 760px), (pointer: coarse) and (max-height: 500px)');
/** A phone held sideways, where the sheet is a column beside the car. */
const SIDE = matchMedia(
	'(max-width: 760px) and (max-height: 500px) and (orientation: landscape), (pointer: coarse) and (max-height: 500px) and (orientation: landscape)',
);
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)');

/** The half detent's height, as a share of the height under the top bar. */
const HALF = 0.56;
/** How far a press moves, in px, before it drags the sheet rather than taps what it pressed. */
const DRAG_PX = 6;
/** How far ahead, in seconds, a released drag's speed carries the sheet when choosing where it settles. */
const FLING_S = 0.18;
/** The sheet's settling move, as app.css times it: rising, and the faster falling. */
const RISE_MS = 320;
const FALL_MS = 220;
/** Where a press drags the sheet: its grip and the head of what it shows. */
const GRAB = '.sheet-grip, .sheet-head, .files-head, .assistant-head';
/**
 * How much of the room the sheet leaves the car's box fills when the car is framed into it: nearly all, since the
 * box's corners lie outside the car seen at an angle, which keeps a margin round the car itself.
 */
const FRAME_FILL = 0.95;

/** The detent each page opens at; the case's own is wherever the engineer left it. */
const PAGE_DETENT = { files: 'half', assistant: 'full', model: 'full', export: 'half' };
const PAGE_TITLE = { files: 'Files', assistant: 'Assistant', model: 'Model', export: 'Export' };

/**
 * work: the area under the top bar; topbar: the bar; legend: the legend's place over the car. parts: the tool's
 * elements the sheet holds, {fields, card, dock, files, assistant, model, exports}, `exports` the export menu
 * ({content, refresh}). pages: for each page but Export, {open(on), shown()}, its owner's state. dock: {tab(),
 * choose(name)}, the dock tab shown and choosing one; counts: {probes(), variants()} for the tabs' labels; active():
 * the case the numbers describe, once its car is on screen.
 */
export function createPhoneLayout({ work, topbar, viewer, legend, parts, pages, dock, counts, active }) {
	let exportShown = false;
	const pageShown = () =>
		Object.keys(PAGE_TITLE).find((name) => (name === 'export' ? exportShown : pages[name].shown())) ?? null;

	function openPage(name) {
		for (const other of Object.keys(pages)) if (other !== name && pages[other].shown()) pages[other].open(false);
		exportShown = name === 'export';
		if (exportShown) parts.exports.refresh();
		if (name !== 'export') pages[name].open(true);
		render();
	}
	function closePage(name) {
		if (name === 'export') exportShown = false;
		else pages[name].open(false);
		render();
	}

	// ---- the top bar's menu: every page, one entry ----------------------------------------------------------------

	const entries = [
		['files', 'folder'],
		['assistant', 'assistant'],
		['model', 'views'],
		['export', 'download'],
	].map(([name, glyph]) =>
		h(
			'button',
			{
				type: 'button',
				role: 'menuitem',
				class: 'menu-item',
				'data-page': name,
				onclick: () => {
					more.close();
					openPage(name);
				},
			},
			icon(glyph),
			h('span', {}, PAGE_TITLE[name]),
		),
	);
	const more = menu(
		h('button', { type: 'button', class: 'icon-button more-button', 'aria-label': 'Menu' }, icon('more')),
		h('div', { class: 'menu-list', role: 'menu' }, entries),
		{ align: 'right' },
	);
	more.el.classList.add('more-menu');
	topbar.append(more.el);

	// ---- the sheet ------------------------------------------------------------------------------------------------

	/**
	 * The case's head: the headline numbers while the sheet is low, the panel's tabs in their place once it is up, and
	 * the field switch under either. The Numbers tab is the sheet's own; the others are the dock's tabs.
	 */
	let numbers = true;
	const headline = h('div', { class: 'sheet-headline' });
	const fieldsSlot = h('div', { class: 'sheet-fields' });
	const tabs = segmented(
		'Panel',
		[
			['numbers', 'Numbers'],
			['section', 'Section'],
			['probes', 'Probes'],
			['variants', 'Variants'],
		],
		(name) => {
			numbers = name === 'numbers';
			if (!numbers) dock.choose(name);
			render();
		},
	);
	tabs.el.classList.add('sheet-tabs');
	const numbersSlot = h('div', { class: 'sheet-numbers' });
	const dockSlot = h('div', { class: 'sheet-dock' });
	const caseHead = h(
		'div',
		{ class: 'sheet-head case-head' },
		headline,
		h('div', { class: 'sheet-tab-row' }, tabs.el),
		fieldsSlot,
	);
	const panes = {
		case: h(
			'div',
			{ class: 'sheet-pane', 'data-pane': 'case' },
			caseHead,
			h('div', { class: 'sheet-body' }, numbersSlot, dockSlot),
		),
		files: h('div', { class: 'sheet-pane', 'data-pane': 'files' }),
		assistant: h('div', { class: 'sheet-pane', 'data-pane': 'assistant' }),
		model: h('div', { class: 'sheet-pane', 'data-pane': 'model' }, pageHead('model')),
		export: h('div', { class: 'sheet-pane', 'data-pane': 'export' }, pageHead('export')),
	};
	const modelSlot = h('div', { class: 'sheet-page' });
	const exportSlot = h('div', { class: 'sheet-page' });
	panes.model.append(modelSlot);
	panes.export.append(exportSlot);
	const grip = h('div', { class: 'sheet-grip', 'aria-hidden': 'true' }, h('i'));
	const inner = h('div', { class: 'sheet-inner' }, grip, Object.values(panes));
	const sheet = h('section', { class: 'sheet', 'aria-label': 'Panel' }, inner);
	work.append(sheet);

	function pageHead(name) {
		return h(
			'div',
			{ class: 'sheet-head page-head' },
			h('h2', { class: 'page-title' }, PAGE_TITLE[name]),
			h(
				'button',
				{
					type: 'button',
					class: 'icon-button',
					'aria-label': `Close ${PAGE_TITLE[name]}`,
					onclick: () => closePage(name),
				},
				icon('close'),
			),
		);
	}

	// ---- moving the tool's elements in and out ---------------------------------------------------------------------

	/** An element that sits in `slot` on a phone and returns, on a desktop, to where it was first found. */
	function movable(el, slot) {
		let home = null;
		return {
			toPhone() {
				if (!home) {
					home = document.createComment('');
					el.before(home);
				}
				if (el.parentNode !== slot) slot.append(el);
			},
			toDesktop() {
				if (home && home.nextSibling !== el) home.after(el);
			},
		};
	}
	const moves = [
		movable(parts.fields, fieldsSlot),
		movable(parts.card, numbersSlot),
		movable(parts.dock, dockSlot),
		movable(parts.files, panes.files),
		movable(parts.assistant, panes.assistant),
		movable(parts.model, modelSlot),
		movable(parts.exports.content, exportSlot),
	];

	// ---- detents ---------------------------------------------------------------------------------------------------

	/** The detent the sheet is at or settling to, and the case's own, kept while a page is open. */
	let detent = 'peek';
	let caseDetent = 'peek';
	/** The pane on show: 'case' or a page; and whether the sheet shows at all. */
	let pane = 'case';
	let shown = false;
	/** The sheet's height on screen, in px, and the drag under way. */
	let height = 0;
	let drag = null;
	let settleTimer = 0;

	const full = () => work.clientHeight;
	function heights() {
		const H = full();
		const head = panes[pane].querySelector('.sheet-head, .files-head, .assistant-head') ?? grip;
		const top = sheet.getBoundingClientRect().top;
		const peek = Math.round(
			head.getBoundingClientRect().bottom - top + parseFloat(getComputedStyle(inner).paddingBottom),
		);
		return { peek, half: Math.max(peek, Math.round(H * HALF)), full: H };
	}

	/** Draw the sheet `px` tall, and its content laid out `content` tall. */
	function place(px, content = px) {
		height = px;
		sheet.style.transform = `translate3d(0, ${full() - px}px, 0)`;
		inner.style.height = `${content}px`;
	}

	/** What the sheet covers of the car, which the car keeps clear of: no more than the half detent. */
	function cover() {
		if (!PHONE.matches || SIDE.matches || !shown) return 0;
		const d = heights();
		return Math.min(d[detent], d.half);
	}
	/**
	 * The car's room: none while the sheet is hidden, 'low' above the low detent, 'raised' above a higher one or beside
	 * the sheet held sideways; marked `data-raised` on the work area for what floats over the car. The car is framed into
	 * the height left below the legend and above the sheet each time its room changes, so it is whole at every detent.
	 */
	let room = null;
	function coverChanged() {
		const bottom = cover();
		const side = PHONE.matches && SIDE.matches;
		const next = !PHONE.matches || !shown ? null : side || detent !== 'peek' ? 'raised' : 'low';
		const frame = next !== null && next !== room && viewer.surface !== null;
		room = next;
		work.toggleAttribute('data-raised', room === 'raised');
		// The legend floats over the top of the car's view, and the car keeps below it while the sheet is out.
		const top = room ? Math.round(legend.getBoundingClientRect().bottom - work.getBoundingClientRect().top + 8) : 0;
		work.style.setProperty('--cover', `${bottom}px`);
		viewer.setInset({ top, bottom }, { frame: frame ? { fill: FRAME_FILL } : null });
	}

	/** Move to a detent: at once, or eased there, the content growing first when it rises and after when it falls. */
	function settle(name, animate = !REDUCED_MOTION.matches) {
		detent = name;
		if (pane === 'case') caseDetent = name;
		clearTimeout(settleTimer);
		if (!PHONE.matches || SIDE.matches) {
			sheet.style.transform = '';
			inner.style.height = '';
			coverChanged();
			return;
		}
		const target = heights()[name];
		const falling = target < height;
		sheet.classList.toggle('settling', animate);
		sheet.classList.toggle('falling', falling);
		place(target, animate && falling ? height : target);
		coverChanged();
		if (animate)
			settleTimer = setTimeout(
				() => {
					place(target);
					sheet.classList.remove('settling');
				},
				falling ? FALL_MS : RISE_MS,
			);
	}

	sheet.addEventListener('pointerdown', (e) => {
		if (SIDE.matches || e.button !== 0 || !e.target.closest(GRAB)) return;
		drag = { id: e.pointerId, y: e.clientY, from: height, moved: false, last: [e.timeStamp, e.clientY], speed: 0 };
	});
	sheet.addEventListener('pointermove', (e) => {
		if (drag?.id !== e.pointerId) return;
		const dy = e.clientY - drag.y;
		if (!drag.moved) {
			if (Math.abs(dy) < DRAG_PX) return;
			drag.moved = true;
			sheet.setPointerCapture(e.pointerId);
			sheet.classList.remove('settling');
			clearTimeout(settleTimer);
		}
		const d = heights();
		let to = drag.from - dy;
		// Past either end the sheet follows at a quarter of the finger's pace.
		if (to > d.full) to = d.full + (to - d.full) / 4;
		if (to < d.peek) to = d.peek - (d.peek - to) / 4;
		const [t, y] = drag.last;
		if (e.timeStamp > t) drag.speed = (y - e.clientY) / ((e.timeStamp - t) / 1000);
		drag.last = [e.timeStamp, e.clientY];
		place(to, Math.max(to, inner.offsetHeight));
	});
	sheet.addEventListener('pointerup', (e) => {
		if (drag?.id !== e.pointerId) return;
		const { moved, speed } = drag;
		drag = null;
		if (moved) {
			const d = heights();
			const aim = height + speed * FLING_S;
			const names = ['peek', 'half', 'full'];
			settle(names.reduce((best, n) => (Math.abs(d[n] - aim) < Math.abs(d[best] - aim) ? n : best)));
			return;
		}
		// A tap on the head, off its controls, raises the sheet to half height, or lowers it from there.
		if (e.target.closest('button, input, a, label, [role="radio"], [role="tab"]')) return;
		settle(detent === 'peek' ? 'half' : 'peek');
	});
	sheet.addEventListener('pointercancel', (e) => {
		if (drag?.id !== e.pointerId) return;
		const moved = drag.moved;
		drag = null;
		if (moved) settle(detent);
	});

	new ResizeObserver(() => {
		if (shown) settle(detent, false);
	}).observe(work);

	// ---- rendering --------------------------------------------------------------------------------------------------

	function renderHeadline(c) {
		const r = c?.result;
		if (!r) {
			headline.replaceChildren();
			return;
		}
		const cfd = r.referenceTotals?.coefficients[0];
		headline.replaceChildren(
			h(
				'span',
				{ class: 'headline-cx' },
				h('span', { class: 'headline-label' }, coefficient('x')),
				h('b', { class: 'num' }, signed(r.totals.coefficients[0], 3)),
				cfd === undefined ? null : h('span', { class: 'headline-cfd num' }, 'CFD ', signed(cfd, 3)),
			),
			h(
				'span',
				{ class: 'headline-cp' },
				h('span', { class: 'headline-label' }, 'Cp'),
				h('i', { class: 'max', 'aria-label': 'max' }),
				h('b', { class: 'num' }, signed(r.totals.cpMax.value, 2)),
				h('i', { class: 'min', 'aria-label': 'min' }),
				h('b', { class: 'num' }, signed(r.totals.cpMin.value, 2)),
			),
		);
	}

	function renderTabs(c) {
		const result = Boolean(c?.result);
		const counted = (label, n) => (result && n ? `${label} (${n})` : label);
		const labels = { probes: counted('Probes', counts.probes()), variants: counted('Variants', counts.variants()) };
		tabs.buttons.forEach((b, i) => {
			const name = ['numbers', 'section', 'probes', 'variants'][i];
			b.textContent = labels[name] ?? b.textContent;
			b.disabled = !result;
		});
		const chosen = numbers ? 'numbers' : dock.tab();
		tabs.show(chosen);
		numbersSlot.hidden = !numbers;
		dockSlot.hidden = numbers;
	}

	/** The bar's tab of the case on show, scrolled into the bar's view when another case comes on show. */
	const caseTabs = topbar.querySelector('.cases');
	let tabInView = null;
	function revealCaseTab() {
		const selected = caseTabs.querySelector('[aria-selected="true"]');
		if (!selected || selected.title === tabInView) return;
		tabInView = selected.title;
		selected.scrollIntoView({ block: 'nearest', inline: 'nearest' });
	}

	/** Bring the sheet up to date: which pane it shows, at which detent, and the case's headline and tabs. */
	function render() {
		if (!PHONE.matches) return;
		const page = pageShown();
		const nextPane = page ?? 'case';
		const nextShown = page !== null || (document.body.dataset.screen === 'case' && active() !== null);
		for (const [name, el] of Object.entries(panes)) el.hidden = name !== nextPane;
		for (const entry of entries) {
			entry.setAttribute('aria-checked', String(entry.dataset.page === page));
			if (entry.dataset.page === 'export') entry.disabled = !active();
		}
		sheet.hidden = !nextShown;
		const c = active();
		renderHeadline(c);
		renderTabs(c);
		revealCaseTab();
		if (nextPane === pane && nextShown === shown) {
			coverChanged();
			return;
		}
		const wasShown = shown;
		pane = nextPane;
		shown = nextShown;
		if (!shown) {
			coverChanged();
			return;
		}
		// A sheet coming up rises from below the screen; one changing pane moves from where it is.
		if (!wasShown) place(0);
		settle(page ? PAGE_DETENT[page] : caseDetent);
	}

	/** A case arriving on screen opens its sheet low, on its numbers. */
	function arrived() {
		numbers = true;
		caseDetent = 'peek';
		if (pane === 'case' && shown) settle('peek');
	}

	function arrange() {
		const phone = PHONE.matches;
		for (const m of moves) (phone ? m.toPhone : m.toDesktop)();
		shown = false;
		pane = 'case';
		render();
		if (!phone) coverChanged();
	}
	PHONE.addEventListener('change', arrange);
	SIDE.addEventListener('change', () => shown && settle(detent, false));
	if (PHONE.matches) arrange();

	return { render, arrived };
}
