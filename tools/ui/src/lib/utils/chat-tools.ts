import { CHAT_TOOLS } from '$lib/constants';
import { ContentPartType, MessageRole, MessageType } from '$lib/enums';
import type {
	ApiChatMessageContentPart,
	ChatMessageTimings,
	DatabaseConversation,
	DatabaseMemory,
	DatabaseMessage
} from '$lib/types';

/** Tokens a request left in the context: the prompt it read plus what it generated. */
export function contextTokens(timings?: ChatMessageTimings | null): number {
	if (!timings) return 0;

	return (timings.prompt_n ?? 0) + (timings.cache_n ?? 0) + (timings.predicted_n ?? 0);
}

/** The line that tells the model the context is filling up and it can compact. */
export function contextReminderText(used: number, total: number): string {
	const percent = Math.round((used / total) * 100);

	return `${CHAT_TOOLS.REMINDER_PREFIX} ${percent}% of the context window is used (${used} of ${total} tokens). You can call compact_conversation to summarize the earlier turns.]`;
}

/** Whether message content already carries a context reminder. */
export function hasContextReminder(
	content: string | ApiChatMessageContentPart[] | undefined
): boolean {
	if (!content) return false;

	if (typeof content === 'string') return content.includes(CHAT_TOOLS.REMINDER_PREFIX);

	return content.some(
		(part) =>
			part.type === ContentPartType.TEXT && !!part.text?.includes(CHAT_TOOLS.REMINDER_PREFIX)
	);
}

/** Message content with a paragraph of text appended, as a string or as one more text part. */
export function appendTextToContent<T extends string | ApiChatMessageContentPart[] | undefined>(
	content: T,
	text: string
): string | ApiChatMessageContentPart[] {
	if (Array.isArray(content)) return [...content, { text, type: ContentPartType.TEXT }];

	return content ? `${content}\n\n${text}` : text;
}

/**
 * A title the model asked for, as one line without surrounding quotes and cut
 * to the length the chat list shows. Empty when nothing usable is left.
 */
export function normalizeConversationTitle(raw: unknown): string {
	if (typeof raw !== 'string') return '';

	const title = raw
		.replace(/\s+/g, ' ')
		.trim()
		.replace(/^["'`]+|["'`]+$/g, '')
		.trim();

	if (title.length <= CHAT_TOOLS.TITLE_MAX_LENGTH) return title;

	return title.slice(0, CHAT_TOOLS.TITLE_MAX_LENGTH).trimEnd();
}

/** A memory as the model wrote it, on one line. Empty when nothing usable is left. */
export function normalizeMemoryContent(raw: unknown): string {
	if (typeof raw !== 'string') return '';

	return raw.replace(/\s+/g, ' ').trim();
}

/** The lowercase words of a search query, without punctuation around them, each once. */
export function queryTerms(query: unknown): string[] {
	if (typeof query !== 'string') return [];

	const terms = query
		.toLowerCase()
		.split(/\s+/)
		.map((term) => term.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ''))
		.filter((term) => term.length > 0);

	return [...new Set(terms)];
}

function termsIn(lowerText: string, terms: string[]): string[] {
	return terms.filter((term) => lowerText.includes(term));
}

/**
 * The memory list a new chat's system prompt carries: the newest memories that
 * fit the limits, listed oldest first with their ids. Empty without memories.
 */
export function formatMemoryPromptBlock(memories: readonly DatabaseMemory[]): string {
	const lines: string[] = [];

	let chars = 0;

	for (const memory of [...memories].sort((a, b) => b.updatedAt - a.updatedAt)) {
		const line = `- [${memory.id}] ${memory.content}`;

		if (lines.length >= CHAT_TOOLS.MEMORY_PROMPT_MAX_ITEMS) break;

		if (chars + line.length + 1 > CHAT_TOOLS.MEMORY_PROMPT_MAX_CHARS) break;

		lines.push(line);
		chars += line.length + 1;
	}

	if (lines.length === 0) return '';

	return [CHAT_TOOLS.MEMORY_PROMPT_HEADER, ...lines.reverse()].join('\n');
}

/**
 * Memories for a query: the ones holding the most query words first, newer
 * first among equals. An empty query lists the newest.
 */
export function rankMemories(
	memories: readonly DatabaseMemory[],
	query: unknown,
	limit: number
): DatabaseMemory[] {
	const terms = queryTerms(query);
	const newest = [...memories].sort((a, b) => b.updatedAt - a.updatedAt);

	if (terms.length === 0) return newest.slice(0, limit);

	return newest
		.map((memory) => ({ hits: termsIn(memory.content.toLowerCase(), terms).length, memory }))
		.filter((entry) => entry.hits > 0)
		.sort((a, b) => b.hits - a.hits)
		.slice(0, limit)
		.map((entry) => entry.memory);
}

/** Text around the first hit of any term on one line, with ellipses where it was cut. */
export function snippetAround(
	text: string,
	terms: string[],
	length: number = CHAT_TOOLS.SNIPPET_CHARS
): string {
	const flat = text.replace(/\s+/g, ' ').trim();

	if (flat.length <= length) return flat;

	const lower = flat.toLowerCase();

	let at = -1;

	for (const term of terms) {
		const index = lower.indexOf(term);

		if (index >= 0 && (at < 0 || index < at)) at = index;
	}

	const start = at < 0 ? 0 : Math.max(0, at - Math.floor(length / 3));
	const end = Math.min(flat.length, start + length);

	return `${start > 0 ? '...' : ''}${flat.slice(start, end)}${end < flat.length ? '...' : ''}`;
}

/** One conversation search_conversations returns. */
export interface ConversationSearchHit {
	conversation_id: string;
	last_active: string;
	snippets: string[];
	title: string;
}

/** The messages a conversation search reads: what the user and the model wrote, and summaries. */
function isSearchable(message: DatabaseMessage): boolean {
	if (message.isSynthetic || !message.content.trim()) return false;

	if (message.type === MessageType.COMPACTION) return true;

	return message.role === MessageRole.USER || message.role === MessageRole.ASSISTANT;
}

function snippetLine(message: DatabaseMessage, terms: string[]): string {
	const label = message.type === MessageType.COMPACTION ? 'summary' : message.role;

	return `${label}: ${snippetAround(message.content, terms)}`;
}

function isoDate(timestamp: number): string {
	return new Date(timestamp).toISOString().slice(0, 10);
}

/**
 * Conversations matching a query, best first. Conversations holding every
 * word, in their messages or title, come first and alone when there are any;
 * otherwise the ones holding the most words, flagged as partial. Among equals
 * more matching messages, then the more recent conversation, rank higher. An
 * empty query lists the most recent conversations with their opening message.
 */
export function searchConversations(
	conversations: readonly DatabaseConversation[],
	messages: readonly DatabaseMessage[],
	query: unknown,
	options: { excludeConvId?: string; limit?: number } = {}
): { hits: ConversationSearchHit[]; partial: boolean } {
	const limit = options.limit ?? CHAT_TOOLS.SEARCH_MAX_RESULTS;
	const terms = queryTerms(query);
	const candidates = conversations.filter((c) => c.id !== options.excludeConvId);
	const byConversation = new Map<string, DatabaseMessage[]>();

	for (const message of messages) {
		if (!isSearchable(message)) continue;

		const list = byConversation.get(message.convId);

		if (list) list.push(message);
		else byConversation.set(message.convId, [message]);
	}

	if (terms.length === 0) {
		const hits = [...candidates]
			.sort((a, b) => b.lastModified - a.lastModified)
			.slice(0, limit)
			.map((conversation) => {
				const opening = (byConversation.get(conversation.id) ?? [])
					.filter((m) => m.role === MessageRole.USER)
					.sort((a, b) => a.timestamp - b.timestamp)[0];

				return {
					conversation_id: conversation.id,
					last_active: isoDate(conversation.lastModified),
					snippets: opening ? [snippetLine(opening, [])] : [],
					title: conversation.name
				};
			});

		return { hits, partial: false };
	}

	const scored = candidates.map((conversation) => {
		const found = new Set(termsIn(conversation.name.toLowerCase(), terms));
		const matching: { hits: number; message: DatabaseMessage }[] = [];

		for (const message of byConversation.get(conversation.id) ?? []) {
			const hits = termsIn(message.content.toLowerCase(), terms);

			if (hits.length === 0) continue;

			for (const term of hits) found.add(term);

			matching.push({ hits: hits.length, message });
		}

		return { conversation, found: found.size, matching };
	});
	const complete = scored.filter((entry) => entry.found === terms.length);
	const pool = complete.length > 0 ? complete : scored.filter((entry) => entry.found > 0);
	const hits = pool
		.sort(
			(a, b) =>
				b.found - a.found ||
				b.matching.length - a.matching.length ||
				b.conversation.lastModified - a.conversation.lastModified
		)
		.slice(0, limit)
		.map(({ conversation, matching }) => ({
			conversation_id: conversation.id,
			last_active: isoDate(conversation.lastModified),
			snippets: [...matching]
				.sort((a, b) => b.hits - a.hits || a.message.timestamp - b.message.timestamp)
				.slice(0, CHAT_TOOLS.SNIPPETS_PER_CONVERSATION)
				.map((entry) => snippetLine(entry.message, terms)),
			title: conversation.name
		}));

	return { hits, partial: complete.length === 0 && hits.length > 0 };
}

/** The tool names an assistant message called, in order. */
function calledTools(message: DatabaseMessage): string[] {
	if (!message.toolCalls) return [];

	try {
		const calls = JSON.parse(message.toolCalls) as { function?: { name?: string } }[];

		return calls.map((call) => call.function?.name ?? '').filter(Boolean);
	} catch {
		return [];
	}
}

function transcriptLine(message: DatabaseMessage, index: number): string | null {
	const text = message.content.trim();

	if (message.type === MessageType.COMPACTION) return text ? `[${index}] summary: ${text}` : null;

	if (message.role === MessageRole.TOOL) {
		const cut =
			text.length > CHAT_TOOLS.READ_TOOL_RESULT_CHARS
				? `${text.slice(0, CHAT_TOOLS.READ_TOOL_RESULT_CHARS)}... (cut)`
				: text;

		return `[${index}] tool result: ${cut}`;
	}

	if (message.role === MessageRole.ASSISTANT) {
		const tools = calledTools(message);
		const called = tools.length > 0 ? ` (called ${tools.join(', ')})` : '';

		return text || called ? `[${index}] assistant${called}: ${text}` : null;
	}

	return text ? `[${index}] ${message.role}: ${text}` : null;
}

/**
 * A conversation branch as a numbered transcript from `offset`, as long as it
 * fits `maxChars`; at least one message, cut to fit, so a long message never
 * stalls the reader. The system prompt is left out. `next_offset` is where to
 * read on, null at the end.
 */
export function renderTranscript(
	branch: readonly DatabaseMessage[],
	offset: number,
	maxChars: number = CHAT_TOOLS.READ_MAX_CHARS
): { from: number; next_offset: number | null; total_messages: number; transcript: string } {
	const messages = branch.filter((m) => m.role !== MessageRole.SYSTEM);
	const from = Math.min(Math.max(0, Math.floor(offset) || 0), messages.length);
	const lines: string[] = [];

	let chars = 0;
	let next = from;

	for (; next < messages.length; next++) {
		const line = transcriptLine(messages[next], next);

		if (line === null) continue;

		if (lines.length > 0 && chars + line.length + 1 > maxChars) break;

		lines.push(line.length > maxChars ? `${line.slice(0, maxChars)}... (cut)` : line);
		chars += line.length + 1;
	}

	return {
		from,
		next_offset: next < messages.length ? next : null,
		total_messages: messages.length,
		transcript: lines.join('\n')
	};
}
