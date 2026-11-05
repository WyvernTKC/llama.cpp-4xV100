import { SPEECH_CHUNK_MAX_CHARS } from '$lib/constants';

/** The part of the Web Speech recognizer the chat form uses (not in the TS DOM lib). */
export interface SpeechRecognitionLike {
	continuous: boolean;
	interimResults: boolean;
	lang: string;
	onend: (() => void) | null;
	onerror: ((event: { error: string }) => void) | null;
	onresult: ((event: { results: SpeechRecognitionResultList }) => void) | null;
	abort(): void;
	start(): void;
	stop(): void;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

const SPEECH_MARKDOWN_RULES: [RegExp, string][] = [
	[/```[\s\S]*?(```|$)/g, '\n'],
	[/!\[[^\]]*\]\([^)]*\)/g, ' '],
	[/\[([^\]]*)\]\([^)]*\)/g, '$1'],
	[/https?:\/\/\S+/g, ' '],
	[/<[^>]+>/g, ' '],
	[/^[ \t]{0,3}#{1,6}[ \t]+/gm, ''],
	[/^[ \t]*>[ \t]?/gm, ''],
	[/^[ \t]*([-*_][ \t]*){3,}$/gm, ''],
	[/^[ \t]*[-*+][ \t]+/gm, ''],
	[/^[ \t|:-]*-{3,}[ \t|:-]*$/gm, ''],
	[/[ \t]*\|[ \t]*/g, ', '],
	[/[*`~]+|__/g, ''],
	[/\p{Extended_Pictographic}|\u200D|\uFE0F/gu, '']
];
// a sentence ends at . ! ? ; followed by a space, so "3.5" or "e.g.x" stay whole
const SPEECH_SENTENCE_REGEX = /[^\n]*?[.!?;](?=\s|$)|[^\n]+/g;

function getSpeechRecognitionCtor(): SpeechRecognitionCtor | undefined {
	if (typeof window === 'undefined') return undefined;

	const w = window as unknown as {
		SpeechRecognition?: SpeechRecognitionCtor;
		webkitSpeechRecognition?: SpeechRecognitionCtor;
	};

	return w.SpeechRecognition ?? w.webkitSpeechRecognition;
}

export function isSpeechRecognitionSupported(): boolean {
	return !!getSpeechRecognitionCtor();
}

export function isSpeechSynthesisSupported(): boolean {
	return (
		typeof window !== 'undefined' &&
		'speechSynthesis' in window &&
		typeof SpeechSynthesisUtterance !== 'undefined'
	);
}

/** One-shot recognizer in the device language: it stops when the user stops talking. */
export function createSpeechRecognition(): SpeechRecognitionLike | null {
	const Ctor = getSpeechRecognitionCtor();

	if (!Ctor) return null;

	const recognition = new Ctor();

	recognition.continuous = false;
	recognition.interimResults = true;
	recognition.lang = navigator.language;

	return recognition;
}

export function speechResultsToText(results: SpeechRecognitionResultList): string {
	let text = '';

	for (let i = 0; i < results.length; i++) {
		text += results[i][0]?.transcript ?? '';
	}

	return text.replace(/\s+/g, ' ').trim();
}

/**
 * Markdown to plain text for read-aloud: drops code blocks, images, URLs,
 * markup and emoji. Each line that has no end mark gets a period, so the
 * voice pauses after headings and list items.
 */
export function markdownToSpeechText(markdown: string): string {
	let text = markdown;

	for (const [pattern, replacement] of SPEECH_MARKDOWN_RULES) {
		text = text.replace(pattern, replacement);
	}

	return text
		.split('\n')
		.map((line) => line.replace(/\s+/g, ' ').replace(/^[\s,]+|[\s,]+$/g, ''))
		.filter((line) => /[\p{L}\p{N}]/u.test(line))
		.map((line) => (/[.!?;:]$/.test(line) ? line : `${line}.`))
		.join('\n');
}

/** Groups sentences into chunks of at most maxChars, split at spaces when a sentence is longer. */
export function splitSpeechText(text: string, maxChars = SPEECH_CHUNK_MAX_CHARS): string[] {
	const chunks: string[] = [];

	let chunk = '';

	for (const match of text.matchAll(SPEECH_SENTENCE_REGEX)) {
		const sentence = match[0].trim();

		if (!sentence) continue;

		if (chunk && chunk.length + 1 + sentence.length > maxChars) {
			chunks.push(chunk);
			chunk = '';
		}

		chunk = chunk ? `${chunk} ${sentence}` : sentence;

		while (chunk.length > maxChars) {
			const space = chunk.lastIndexOf(' ', maxChars);
			const cut = space > 0 ? space : maxChars;

			chunks.push(chunk.slice(0, cut));
			chunk = chunk.slice(cut).trim();
		}
	}

	if (chunk) chunks.push(chunk);

	return chunks;
}
