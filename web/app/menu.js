// A button that opens a small floating panel under it. One is open at a time; Escape or a press outside closes it.

import { h } from './dom.js';

let open = null;

function close() {
	if (!open) return;
	open.panel.hidden = true;
	open.button.setAttribute('aria-expanded', 'false');
	open = null;
}

addEventListener('pointerdown', (e) => {
	if (open && !open.root.contains(e.target)) close();
});
addEventListener('keydown', (e) => {
	if (e.key === 'Escape') close();
});

/** `button` toggles `content`; the panel aligns to the button's left or right edge. */
export function menu(button, content, { align = 'left' } = {}) {
	const panel = h('div', { class: `popover popover-${align}`, hidden: true, role: 'dialog' }, content);
	const root = h('div', { class: 'menu' }, button, panel);
	const entry = { root, button, panel };
	button.setAttribute('aria-haspopup', 'true');
	button.setAttribute('aria-expanded', 'false');
	button.addEventListener('click', () => {
		const wasOpen = open === entry;
		close();
		if (wasOpen) return;
		panel.hidden = false;
		button.setAttribute('aria-expanded', 'true');
		open = entry;
	});
	return { el: root, close };
}
