/**
 * ChatHistoryService - search and read past conversations for the chat tools
 *
 * search_conversations matches words over every conversation's messages and
 * titles; read_conversation returns a conversation's current branch as a
 * transcript, part by part. Read only, straight from IndexedDB.
 */

import { CHAT_TOOLS } from '#lib/constants/index.ts';
import { DatabaseService } from '#lib/services/database.service.ts';
import type { DatabaseConversation } from '#lib/types/database.js';
import {
	type ConversationSearchHit,
	filterByLeafNodeId,
	renderTranscript,
	searchConversations
} from '#lib/utils/index.ts';

export class ChatHistoryService {
	/**
	 * A conversation by id, or by the start of one when that start is unique.
	 * @returns The conversation, or null when no single one matched
	 */
	static async find(id: string): Promise<DatabaseConversation | null> {
		const wanted = id.trim();

		if (!wanted) return null;

		const exact = await DatabaseService.getConversation(wanted);

		if (exact) return exact;

		const matches = (await DatabaseService.getAllConversations()).filter((c) =>
			c.id.startsWith(wanted)
		);

		return matches.length === 1 ? matches[0] : null;
	}

	/**
	 * A conversation's current branch as a transcript from `offset`.
	 * @returns null when the conversation does not exist
	 */
	static async read(
		id: string,
		offset: number
	): Promise<
		| (ReturnType<typeof renderTranscript> & {
				conversation_id: string;
				title: string;
		  })
		| null
	> {
		const conversation = await this.find(id);

		if (!conversation) return null;

		const messages = await DatabaseService.getConversationMessages(conversation.id);
		const branch = filterByLeafNodeId(messages, conversation.currNode ?? '', false);

		return {
			conversation_id: conversation.id,
			title: conversation.name,
			...renderTranscript(branch, offset, CHAT_TOOLS.READ_MAX_CHARS)
		};
	}

	/** Conversations other than `excludeConvId` matching the query, best first. */
	static async search(
		query: unknown,
		excludeConvId?: string
	): Promise<{ hits: ConversationSearchHit[]; partial: boolean }> {
		const [conversations, messages] = await Promise.all([
			DatabaseService.getAllConversations(),
			DatabaseService.getAllMessages()
		]);

		return searchConversations(conversations, messages, query, { excludeConvId });
	}
}
