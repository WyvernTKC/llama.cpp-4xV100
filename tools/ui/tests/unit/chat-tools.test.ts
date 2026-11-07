import { buildChatToolDefinitions, CHAT_TOOL_NAMES, CHAT_TOOLS, TOOL_UI } from '$lib/constants';
import { BuiltInTool, ContentPartType, MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';
import {
	appendTextToContent,
	contextReminderText,
	contextTokens,
	countSteps,
	hasContextReminder,
	normalizeConversationTitle
} from '$lib/utils';
import { describe, expect, it } from 'vitest';

function makeMessage(
	id: string,
	role: MessageRole,
	type: MessageType = MessageType.TEXT
): DatabaseMessage {
	return {
		children: [],
		content: id,
		convId: 'conv',
		id,
		parent: null,
		role,
		timestamp: 0,
		type
	} as DatabaseMessage;
}

describe('chat tool definitions', () => {
	it('offers the three chat tools, each with a tool-call label', () => {
		const names = buildChatToolDefinitions().map((t) => t.function.name);

		expect(names).toEqual([...CHAT_TOOL_NAMES]);

		for (const name of names) {
			expect(TOOL_UI[name as BuiltInTool]).toBeDefined();
		}
	});

	it('requires a title for rename_conversation', () => {
		const rename = buildChatToolDefinitions().find(
			(t) => t.function.name === BuiltInTool.BROWSER_RENAME_CONVERSATION
		);

		expect(rename?.function.parameters?.required).toEqual(['title']);
	});
});

describe('contextTokens', () => {
	it('adds the prompt, the cached prompt and the generated tokens', () => {
		expect(contextTokens({ cache_n: 900, predicted_n: 50, prompt_n: 50 })).toBe(1000);
	});

	it('is 0 without timings', () => {
		expect(contextTokens(undefined)).toBe(0);
		expect(contextTokens({})).toBe(0);
	});
});

describe('context reminder', () => {
	it('states the fill and names compact_conversation', () => {
		const text = contextReminderText(850, 1000);

		expect(text.startsWith(CHAT_TOOLS.REMINDER_PREFIX)).toBe(true);
		expect(text).toContain('85%');
		expect(text).toContain('850 of 1000 tokens');
		expect(text).toContain('compact_conversation');
	});

	it('is recognized in string and in part content', () => {
		const text = contextReminderText(850, 1000);

		expect(hasContextReminder(`result\n\n${text}`)).toBe(true);
		expect(hasContextReminder([{ text, type: ContentPartType.TEXT }])).toBe(true);
		expect(hasContextReminder('plain result')).toBe(false);
		expect(hasContextReminder(undefined)).toBe(false);
	});
});

describe('appendTextToContent', () => {
	it('appends a paragraph to string content', () => {
		expect(appendTextToContent('result', 'note')).toBe('result\n\nnote');
		expect(appendTextToContent('', 'note')).toBe('note');
	});

	it('appends a text part to part content and leaves the original alone', () => {
		const parts = [{ image_url: { url: 'data:' }, type: ContentPartType.IMAGE_URL }];
		const result = appendTextToContent(parts, 'note');

		expect(result).toEqual([...parts, { text: 'note', type: ContentPartType.TEXT }]);
		expect(parts).toHaveLength(1);
	});
});

describe('normalizeConversationTitle', () => {
	it('makes one line without surrounding quotes', () => {
		expect(normalizeConversationTitle('  "Trip\n  to   Rome"  ')).toBe('Trip to Rome');
	});

	it('cuts long titles to the chat list length', () => {
		const title = normalizeConversationTitle('word '.repeat(40));

		expect(title.length).toBeLessThanOrEqual(CHAT_TOOLS.TITLE_MAX_LENGTH);
		expect(title.endsWith(' ')).toBe(false);
	});

	it('is empty for nothing usable', () => {
		expect(normalizeConversationTitle('  ""  ')).toBe('');
		expect(normalizeConversationTitle(42)).toBe('');
		expect(normalizeConversationTitle(undefined)).toBe('');
	});
});

describe('countSteps', () => {
	it('counts user messages and assistant turns, not tool results, prefix or compaction nodes', () => {
		const messages = [
			makeMessage('s', MessageRole.SYSTEM, MessageType.SYSTEM),
			makeMessage('c', MessageRole.USER, MessageType.COMPACTION),
			makeMessage('u1', MessageRole.USER),
			makeMessage('a1', MessageRole.ASSISTANT),
			makeMessage('t1', MessageRole.TOOL),
			makeMessage('t2', MessageRole.TOOL),
			makeMessage('a2', MessageRole.ASSISTANT),
			makeMessage('u2', MessageRole.USER)
		];

		expect(countSteps(messages)).toBe(4);
	});
});
