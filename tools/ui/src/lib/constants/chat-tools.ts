import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

// Chat tools let the model manage its own conversation: compact it, check how
// full the context is, and rename it. They run in the browser like get_datetime.
export const CHAT_TOOLS = {
	/** Context fill, in percent, at which the model is reminded it can compact. */
	DEFAULT_REMINDER_PERCENT: 80,
	/** Opening of the reminder line, also how a stored reminder is recognized. */
	REMINDER_PREFIX: '[Context reminder:',
	TITLE_MAX_LENGTH: 80
} as const;

export const CHAT_TOOL_NAMES: readonly string[] = [
	BuiltInTool.BROWSER_COMPACT_CONVERSATION,
	BuiltInTool.BROWSER_GET_CONTEXT_USAGE,
	BuiltInTool.BROWSER_RENAME_CONVERSATION
];

// The definitions never depend on the conversation state, so the tool list
// stays the same on every request and the server keeps its prompt cache.
export function buildChatToolDefinitions(): OpenAIToolDefinition[] {
	return [
		{
			function: {
				description:
					'Summarizes the earlier part of this conversation to free up context. The summary replaces the older turns in your context once the current reply ends, the most recent steps stay as written, and the user still sees the full history. Call it when get_context_usage shows the context filling up or when the user asks for it, then finish your reply normally.',
				name: BuiltInTool.BROWSER_COMPACT_CONVERSATION,
				parameters: { properties: {}, required: [], type: JsonSchemaType.OBJECT }
			},
			type: ToolCallType.FUNCTION
		},
		{
			function: {
				description:
					'Returns how full the context window of this conversation is: tokens used, context size, percent used, and whether compact_conversation can run.',
				name: BuiltInTool.BROWSER_GET_CONTEXT_USAGE,
				parameters: { properties: {}, required: [], type: JsonSchemaType.OBJECT }
			},
			type: ToolCallType.FUNCTION
		},
		{
			function: {
				description:
					'Sets the title of this conversation in the chat list. Use a short, specific title of a few words.',
				name: BuiltInTool.BROWSER_RENAME_CONVERSATION,
				parameters: {
					properties: {
						title: { description: 'The new title, a few words', type: JsonSchemaType.STRING }
					},
					required: ['title'],
					type: JsonSchemaType.OBJECT
				}
			},
			type: ToolCallType.FUNCTION
		}
	];
}
