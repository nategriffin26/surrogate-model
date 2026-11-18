// How the assistant panel reads the tables in a reply (web/assistant/markdown.js). Run with:
// deno test --allow-read tests/assistant_markdown.test.js
import { deepStrictEqual as assertEquals } from 'node:assert/strict';
import { tableCells } from '../web/assistant/markdown.js';

Deno.test('an escaped pipe in a cell is a literal | and does not split the cell', () => {
	// The header of a reply to "Raise the car 20 mm and show me where the pressure changes most".
	assertEquals(tableCells('| Where | Area with \\|ΔCp\\| > 0.1 | Max \\|ΔCp\\| |'), [
		'Where',
		'Area with |ΔCp| > 0.1',
		'Max |ΔCp|',
	]);
});

Deno.test('a plain row splits at each pipe, with or without the outer ones', () => {
	assertEquals(tableCells('| run 1 | 0.266 |'), ['run 1', '0.266']);
	assertEquals(tableCells('run 1 | 0.266'), ['run 1', '0.266']);
});
