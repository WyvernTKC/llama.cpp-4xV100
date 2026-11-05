import { markdownToSpeechText, splitSpeechText } from '$lib/utils/speech';
import { describe, expect, it } from 'vitest';

describe('markdownToSpeechText', () => {
	it('drops markup and code blocks, and ends each line with a period', () => {
		const markdown = [
			'# Title',
			'',
			'Some **bold** and `code` text.',
			'',
			'```js',
			'const x = 1;',
			'```',
			'',
			'- item one',
			'- item two',
			'',
			'See [the docs](https://example.com) here!'
		].join('\n');

		expect(markdownToSpeechText(markdown)).toBe(
			['Title.', 'Some bold and code text.', 'item one.', 'item two.', 'See the docs here!'].join(
				'\n'
			)
		);
	});

	it('reads table rows as comma lists and skips separator rows and rules', () => {
		expect(markdownToSpeechText('| a | b |\n|---|:---:|\n| 1 | 2 |\n\n---\n\nend')).toBe(
			'a, b.\n1, 2.\nend.'
		);
	});

	it('drops emoji, images, bare URLs and an unclosed code block', () => {
		const smile = String.fromCodePoint(0x1f600);

		expect(
			markdownToSpeechText(`Done ${smile}\n![img](a.png)\nhttps://example.com\n\`\`\`\nfoo(`)
		).toBe('Done.');
	});
});

describe('splitSpeechText', () => {
	it('keeps short text in one chunk', () => {
		expect(splitSpeechText('One. Two.')).toEqual(['One. Two.']);
	});

	it('splits at sentence ends and spaces, not inside numbers', () => {
		expect(splitSpeechText('Pi is 3.14 today. Next.', 12)).toEqual(['Pi is 3.14', 'today. Next.']);
	});

	it('cuts a word longer than the limit', () => {
		expect(splitSpeechText('a'.repeat(25), 10)).toEqual([
			'a'.repeat(10),
			'a'.repeat(10),
			'a'.repeat(5)
		]);
	});

	it('never returns a chunk over the limit', () => {
		const text = Array.from({ length: 40 }, (_, i) => `Sentence number ${i} is here.`).join(' ');
		const chunks = splitSpeechText(text, 50);

		expect(chunks.every((chunk) => chunk.length <= 50)).toBe(true);
		expect(chunks.join(' ')).toBe(text);
	});
});
