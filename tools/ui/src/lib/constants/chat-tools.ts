import { STORAGE_APP_NAME } from './storage.constants';
import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

// Chat tools let the model manage its own conversations: compact one, check
// how full the context is, rename, search and read past chats, start a new
// chat, and keep memories across chats. They run in the browser like
// get_datetime.
export const CHAT_TOOLS = {
	/** Context fill, in percent, at which the model is reminded it can compact. */
	DEFAULT_REMINDER_PERCENT: 80,
	/** Longest memory save_memory accepts, in characters. */
	MEMORY_MAX_LENGTH: 500,
	/** Opening of the memory list a new chat's system prompt carries. */
	MEMORY_PROMPT_HEADER:
		'Memories saved in earlier conversations (search_memories finds more, save_memory adds one, delete_memory removes one by id):',
	/** Most characters the memory list adds to a new chat's system prompt. */
	MEMORY_PROMPT_MAX_CHARS: 4000,
	/** Most memories a new chat's system prompt lists, the newest ones. */
	MEMORY_PROMPT_MAX_ITEMS: 40,
	/** Characters read_conversation returns per call. */
	READ_MAX_CHARS: 12000,
	/** Characters of each tool result read_conversation keeps. */
	READ_TOOL_RESULT_CHARS: 400,
	/** Opening of the reminder line, also how a stored reminder is recognized. */
	REMINDER_PREFIX: '[Context reminder:',
	/** Most conversations search_conversations returns. */
	SEARCH_MAX_RESULTS: 10,
	/** Most memories search_memories returns. */
	SEARCH_MEMORIES_MAX_RESULTS: 20,
	/** Characters of context around a search hit. */
	SNIPPET_CHARS: 160,
	/** Snippets search_conversations returns per conversation. */
	SNIPPETS_PER_CONVERSATION: 2,
	TITLE_MAX_LENGTH: 80
} as const;

/** Memories live in their own database, so the conversation schema never changes for them. */
export const MEMORY_IDXDB = {
	NAME: `${STORAGE_APP_NAME}Memories`,
	SCHEMA: 'id, updatedAt',
	TABLE: 'memories'
} as const;

export const CHAT_TOOL_NAMES: readonly string[] = [
	BuiltInTool.BROWSER_COMPACT_CONVERSATION,
	BuiltInTool.BROWSER_DELETE_MEMORY,
	BuiltInTool.BROWSER_GET_CONTEXT_USAGE,
	BuiltInTool.BROWSER_NEW_CONVERSATION,
	BuiltInTool.BROWSER_READ_CONVERSATION,
	BuiltInTool.BROWSER_RENAME_CONVERSATION,
	BuiltInTool.BROWSER_SAVE_MEMORY,
	BuiltInTool.BROWSER_SEARCH_CONVERSATIONS,
	BuiltInTool.BROWSER_SEARCH_MEMORIES
];

function tool(
	name: BuiltInTool,
	description: string,
	properties: Record<string, unknown> = {},
	required: string[] = []
): OpenAIToolDefinition {
	return {
		function: {
			description,
			name,
			parameters: { properties, required, type: JsonSchemaType.OBJECT }
		},
		type: ToolCallType.FUNCTION
	};
}

// The definitions never depend on the conversation state, so the tool list
// stays the same on every request and the server keeps its prompt cache.
export function buildChatToolDefinitions(): OpenAIToolDefinition[] {
	return [
		tool(
			BuiltInTool.BROWSER_COMPACT_CONVERSATION,
			'Summarizes the earlier part of this conversation to free up context. The summary replaces the older turns in your context once the current reply ends, the most recent steps stay as written, and the user still sees the full history. Call it when get_context_usage shows the context filling up or when the user asks for it, then finish your reply normally.'
		),
		tool(
			BuiltInTool.BROWSER_GET_CONTEXT_USAGE,
			'Returns how full the context window of this conversation is: tokens used, context size, percent used, and whether compact_conversation can run.'
		),
		tool(
			BuiltInTool.BROWSER_RENAME_CONVERSATION,
			'Sets the title of this conversation in the chat list. Use a short, specific title of a few words.',
			{ title: { description: 'The new title, a few words', type: JsonSchemaType.STRING } },
			['title']
		),
		tool(
			BuiltInTool.BROWSER_SEARCH_CONVERSATIONS,
			"Searches the user's other conversations in this browser for words in their messages and titles. Returns the best matching conversations with their id, title, date and short snippets. An empty query lists the most recent conversations. Use read_conversation to read one.",
			{
				query: {
					description: 'Words to look for; conversations holding all of them rank first',
					type: JsonSchemaType.STRING
				}
			},
			['query']
		),
		tool(
			BuiltInTool.BROWSER_READ_CONVERSATION,
			'Reads another conversation as a transcript, part by part. Pass the id from search_conversations; when the result has next_offset, call again with it to read on.',
			{
				conversation_id: {
					description: 'The conversation id from search_conversations',
					type: JsonSchemaType.STRING
				},
				offset: {
					description: 'Message to start from, 0 for the beginning',
					type: JsonSchemaType.NUMBER
				}
			},
			['conversation_id']
		),
		tool(
			BuiltInTool.BROWSER_NEW_CONVERSATION,
			'Starts a new conversation in a new tab, for example to move a separate topic out of this one. The first message is left in its input box for the user to review and send; nothing is sent by this call.',
			{
				first_message: {
					description: 'The message to put in the new chat, written as the user would send it',
					type: JsonSchemaType.STRING
				},
				title: { description: 'Title of the new chat, a few words', type: JsonSchemaType.STRING }
			},
			['title']
		)
	];
}

export function buildMemoryToolDefinitions(): OpenAIToolDefinition[] {
	return [
		tool(
			BuiltInTool.BROWSER_SAVE_MEMORY,
			'Saves a short fact to remember in later conversations, such as a preference of the user or a decision. New chats list the saved memories in their system prompt. Keep each memory to one short, self-contained sentence, and do not save what is only relevant to this conversation.',
			{ content: { description: 'The fact to remember', type: JsonSchemaType.STRING } },
			['content']
		),
		tool(
			BuiltInTool.BROWSER_SEARCH_MEMORIES,
			'Searches the saved memories for words. An empty query lists the most recent ones. Returns each memory with its id.',
			{ query: { description: 'Words to look for', type: JsonSchemaType.STRING } },
			['query']
		),
		tool(
			BuiltInTool.BROWSER_DELETE_MEMORY,
			'Deletes a saved memory that is wrong or no longer true, by the id that search_memories or the memory list shows.',
			{ id: { description: 'The memory id', type: JsonSchemaType.STRING } },
			['id']
		)
	];
}
