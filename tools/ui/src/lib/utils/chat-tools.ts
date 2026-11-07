import { CHAT_TOOLS } from '$lib/constants';
import { ContentPartType } from '$lib/enums';
import type { ApiChatMessageContentPart, ChatMessageTimings } from '$lib/types';

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
