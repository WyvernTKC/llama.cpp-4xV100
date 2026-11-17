import {
	buildChatToolDefinitions,
	buildMemoryToolDefinitions,
	CHAT_TOOL_NAMES,
	CHAT_TOOLS,
	TOOL_UI
} from '#lib/constants/index.ts';
import { BuiltInTool, ContentPartType, MessageRole, MessageType } from '#lib/enums/index.ts';
import type { DatabaseConversation, DatabaseMemory, DatabaseMessage } from '#lib/types/index.ts';
import {
	appendTextToContent,
	contextReminderText,
	contextTokens,
	countSteps,
	formatMemoryPromptBlock,
	hasContextReminder,
	normalizeConversationTitle,
	normalizeMemoryContent,
	queryTerms,
	rankMemories,
	renderTranscript,
	searchConversations,
	snippetAround
} from '#lib/utils/index.ts';
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
	it('covers every chat tool name once, each with a tool-call label', () => {
		const names = [...buildChatToolDefinitions(), ...buildMemoryToolDefinitions()].map(
			(t) => t.function.name
		);

		expect([...names].sort()).toEqual([...CHAT_TOOL_NAMES].sort());

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

function memory(id: string, content: string, updatedAt: number): DatabaseMemory {
	return { content, createdAt: updatedAt, id, updatedAt };
}

describe('memories', () => {
	it('keeps a memory on one line and drops non-text', () => {
		expect(normalizeMemoryContent('  User   likes\n metric units ')).toBe(
			'User likes metric units'
		);
		expect(normalizeMemoryContent(7)).toBe('');
	});

	it('lists the newest memories in the system prompt block, oldest first', () => {
		const block = formatMemoryPromptBlock([
			memory('a', 'first', 1),
			memory('c', 'third', 3),
			memory('b', 'second', 2)
		]);

		expect(block.split('\n')).toEqual([
			CHAT_TOOLS.MEMORY_PROMPT_HEADER,
			'- [a] first',
			'- [b] second',
			'- [c] third'
		]);
		expect(formatMemoryPromptBlock([])).toBe('');
	});

	it('caps the block at the item limit, keeping the newest', () => {
		const many = Array.from({ length: CHAT_TOOLS.MEMORY_PROMPT_MAX_ITEMS + 5 }, (_, i) =>
			memory(`m${i}`, `fact ${i}`, i)
		);
		const lines = formatMemoryPromptBlock(many).split('\n').slice(1);

		expect(lines).toHaveLength(CHAT_TOOLS.MEMORY_PROMPT_MAX_ITEMS);
		expect(lines[0]).toBe('- [m5] fact 5');
	});

	it('ranks memories by matching words, newer first among equals', () => {
		const all = [
			memory('a', 'User prefers metric units', 1),
			memory('b', 'User lives in Brisbane', 2),
			memory('c', 'Metric units for the Brisbane trip', 3)
		];

		expect(rankMemories(all, 'metric brisbane', 10).map((m) => m.id)).toEqual(['c', 'b', 'a']);
		expect(rankMemories(all, '', 2).map((m) => m.id)).toEqual(['c', 'b']);
		expect(rankMemories(all, 'nothing', 10)).toEqual([]);
	});
});

describe('queryTerms and snippets', () => {
	it('splits a query into lowercase words without punctuation', () => {
		expect(queryTerms('  "Rome", trip!  rome ')).toEqual(['rome', 'trip']);
		expect(queryTerms(undefined)).toEqual([]);
	});

	it('cuts a snippet around the first hit', () => {
		const text = `${'a '.repeat(200)}needle ${'b '.repeat(200)}`;
		const snippet = snippetAround(text, ['needle'], 60);

		expect(snippet.startsWith('...')).toBe(true);
		expect(snippet.endsWith('...')).toBe(true);
		expect(snippet).toContain('needle');
		expect(snippetAround('short text', ['x'], 60)).toBe('short text');
	});
});

function conversation(id: string, name: string, lastModified: number): DatabaseConversation {
	return { currNode: null, id, lastModified, name };
}

function chatMessage(
	id: string,
	convId: string,
	role: MessageRole,
	content: string,
	timestamp = 0,
	extra: Partial<DatabaseMessage> = {}
): DatabaseMessage {
	return {
		children: [],
		content,
		convId,
		id,
		parent: null,
		role,
		timestamp,
		type: MessageType.TEXT,
		...extra
	} as DatabaseMessage;
}

describe('searchConversations', () => {
	const conversations = [
		conversation('c1', 'Rome trip', 10),
		conversation('c2', 'Pasta recipes', 20),
		conversation('c3', 'Current chat', 30)
	];
	const messages = [
		chatMessage('m1', 'c1', MessageRole.USER, 'Plan three days in Rome in May', 1),
		chatMessage('m2', 'c1', MessageRole.ASSISTANT, 'Day one: the Colosseum and the Forum', 2),
		chatMessage('m3', 'c2', MessageRole.USER, 'A carbonara like they make in Rome', 3),
		chatMessage('m4', 'c3', MessageRole.USER, 'Rome Colosseum tickets', 4),
		chatMessage('m5', 'c2', MessageRole.TOOL, 'Colosseum tool output', 5)
	];

	it('ranks conversations holding every word first and leaves out the current one', () => {
		const { hits, partial } = searchConversations(conversations, messages, 'rome colosseum', {
			excludeConvId: 'c3'
		});

		expect(partial).toBe(false);
		expect(hits.map((h) => h.conversation_id)).toEqual(['c1']);
		expect(hits[0].title).toBe('Rome trip');
		expect(hits[0].snippets.length).toBeGreaterThan(0);
	});

	it('falls back to partial matches, ignoring tool results', () => {
		const { hits, partial } = searchConversations(conversations, messages, 'carbonara colosseum', {
			excludeConvId: 'c3'
		});

		// one word each, one matching message each: the more recent conversation first
		expect(partial).toBe(true);
		expect(hits.map((h) => h.conversation_id)).toEqual(['c2', 'c1']);
	});

	it('lists the most recent conversations for an empty query', () => {
		const { hits } = searchConversations(conversations, messages, '', { excludeConvId: 'c3' });

		expect(hits.map((h) => h.conversation_id)).toEqual(['c2', 'c1']);
		expect(hits[1].snippets).toEqual(['user: Plan three days in Rome in May']);
	});
});

describe('renderTranscript', () => {
	const branch = [
		chatMessage('s', 'c', MessageRole.SYSTEM, 'system prompt', 0),
		chatMessage('u', 'c', MessageRole.USER, 'hello', 1),
		chatMessage('a', 'c', MessageRole.ASSISTANT, '', 2, {
			toolCalls: JSON.stringify([{ function: { arguments: '{}', name: 'get_datetime' }, id: 't' }])
		}),
		chatMessage('t', 'c', MessageRole.TOOL, 'x'.repeat(1000), 3),
		chatMessage('a2', 'c', MessageRole.ASSISTANT, 'It is noon.', 4)
	];

	it('numbers the messages, names called tools and cuts long tool results', () => {
		const read = renderTranscript(branch, 0);
		const lines = read.transcript.split('\n');

		expect(read.total_messages).toBe(4);
		expect(read.next_offset).toBeNull();
		expect(lines[0]).toBe('[0] user: hello');
		expect(lines[1]).toBe('[1] assistant (called get_datetime): ');
		expect(lines[2].endsWith('... (cut)')).toBe(true);
		expect(lines[3]).toBe('[3] assistant: It is noon.');
	});

	it('pages by size and reads on from next_offset', () => {
		const first = renderTranscript(branch, 0, 60);

		expect(first.next_offset).not.toBeNull();

		const rest = renderTranscript(branch, first.next_offset ?? 0, 10000);

		expect(rest.from).toBe(first.next_offset);
		expect(rest.transcript.endsWith('[3] assistant: It is noon.')).toBe(true);
	});
});
