import { STORAGE_APP_NAME } from './storage.constants';
import { AgentTaskStatus, BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

// Agent tools give the model ways to pace and organize longer work: wait,
// ask the user, keep a task list, notify, wake itself up later and hand a
// task to a sub-agent. They run in the browser like the chat tools.
export const AGENT_TOOLS = {
	/** Most options ask_user shows. */
	ASK_MAX_OPTIONS: 6,
	/** Longest option label ask_user keeps, in characters. */
	ASK_OPTION_MAX_LENGTH: 120,
	/** Title of a notification the model gives none. */
	NOTIFY_DEFAULT_TITLE: 'llama.cpp',
	/** Longest notification text, in characters. */
	NOTIFY_MAX_LENGTH: 300,
	/** Longest sleep, in seconds. */
	SLEEP_MAX_SECONDS: 3600,
	/** How often a sleep checks for a user message, in milliseconds. */
	SLEEP_POLL_MS: 250,
	/** Characters of a sub-agent's final answer returned to the caller. */
	SUBAGENT_ANSWER_MAX_CHARS: 20000,
	/** Opening of a sub-agent chat's title. */
	SUBAGENT_TITLE_PREFIX: 'Sub-agent: ',
	/** Longest task text, in characters. */
	TASK_MAX_LENGTH: 200,
	/** Most tasks update_tasks keeps. */
	TASKS_MAX: 30,
	/** Longest wake-up delay, in minutes. */
	WAKEUP_MAX_MINUTES: 1440,
	/** Opening of the message a wake-up sends, also how one is recognized. */
	WAKEUP_PREFIX: '[Wake-up]',
	/** How often a due wake-up retries while its chat is busy, in milliseconds. */
	WAKEUP_RETRY_MS: 5000,
	/** Delay before a wake-up that fell due while the page was closed fires, in milliseconds. */
	WAKEUP_STARTUP_DELAY_MS: 10000,
	/** localStorage key of the scheduled wake-ups. */
	WAKEUP_STORAGE_KEY: `${STORAGE_APP_NAME}.wakeups`
} as const;

export const AGENT_TOOL_NAMES: readonly string[] = [
	BuiltInTool.BROWSER_ASK_USER,
	BuiltInTool.BROWSER_NOTIFY,
	BuiltInTool.BROWSER_RUN_SUBAGENT,
	BuiltInTool.BROWSER_SCHEDULE_WAKEUP,
	BuiltInTool.BROWSER_SLEEP,
	BuiltInTool.BROWSER_UPDATE_TASKS
];

/**
 * Tools a sub-agent chat does not get: it works alone and reports back, so it
 * neither starts more agents or chats, nor waits on the user, nor wakes
 * itself up after its answer has gone back.
 */
export const SUBAGENT_DISABLED_TOOLS: readonly string[] = [
	BuiltInTool.BROWSER_ASK_USER,
	BuiltInTool.BROWSER_NEW_CONVERSATION,
	BuiltInTool.BROWSER_RUN_SUBAGENT,
	BuiltInTool.BROWSER_SCHEDULE_WAKEUP
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

// The definitions never depend on state, so the tool list stays the same on
// every request and the server keeps its prompt cache.
export function buildAgentToolDefinitions(): OpenAIToolDefinition[] {
	return [
		tool(
			BuiltInTool.BROWSER_SLEEP,
			`Waits a number of seconds before you go on, for example to give a running job time to finish before you check on it again. The wait ends early when the user sends a message. At most ${AGENT_TOOLS.SLEEP_MAX_SECONDS} seconds.`,
			{
				reason: {
					description: 'What you are waiting for, shown to the user',
					type: JsonSchemaType.STRING
				},
				seconds: { description: 'How long to wait, in seconds', type: JsonSchemaType.NUMBER }
			},
			['seconds']
		),
		tool(
			BuiltInTool.BROWSER_ASK_USER,
			`Asks the user a question and waits for the answer. Give 2 to ${AGENT_TOOLS.ASK_MAX_OPTIONS} short options to pick from; the user can also type an answer of their own. Use it when you need the user to decide something before you can go on, not for what you can find out yourself.`,
			{
				options: {
					description: 'The answers to offer, a few words each',
					items: { type: JsonSchemaType.STRING },
					type: JsonSchemaType.ARRAY
				},
				question: { description: 'The question, one or two sentences', type: JsonSchemaType.STRING }
			},
			['question', 'options']
		),
		tool(
			BuiltInTool.BROWSER_UPDATE_TASKS,
			'Keeps a task list for this conversation, shown to the user above the input box. Use it for work of three or more steps. Pass the whole list every time, in order: mark a task in_progress when you start it and completed as soon as it is done, with one task in_progress at a time.',
			{
				tasks: {
					description: 'Every task of the list, in order',
					items: {
						properties: {
							content: {
								description: 'The task, a short imperative phrase',
								type: JsonSchemaType.STRING
							},
							status: {
								enum: [
									AgentTaskStatus.PENDING,
									AgentTaskStatus.IN_PROGRESS,
									AgentTaskStatus.COMPLETED
								],
								type: JsonSchemaType.STRING
							}
						},
						required: ['content', 'status'],
						type: JsonSchemaType.OBJECT
					},
					type: JsonSchemaType.ARRAY
				}
			},
			['tasks']
		),
		tool(
			BuiltInTool.BROWSER_NOTIFY,
			'Shows the user a notification that also reaches them when this page is in the background, for example when a long task is done or you need their input. One short sentence.',
			{
				message: { description: 'The notification text', type: JsonSchemaType.STRING },
				title: { description: 'A short title, optional', type: JsonSchemaType.STRING }
			},
			['message']
		),
		tool(
			BuiltInTool.BROWSER_SCHEDULE_WAKEUP,
			`Continues this conversation later: once the given minutes have passed and your reply has ended, your note comes back to you in this chat as a message starting with ${AGENT_TOOLS.WAKEUP_PREFIX}, and you go on from there. Use it to check on long-running work. A conversation has one wake-up: scheduling again replaces it and 0 minutes cancels it. It fires only while the page is open. At most ${AGENT_TOOLS.WAKEUP_MAX_MINUTES} minutes.`,
			{
				minutes: { description: 'Minutes from now, 0 to cancel', type: JsonSchemaType.NUMBER },
				note: {
					description: 'What to do when you wake up, written to yourself',
					type: JsonSchemaType.STRING
				}
			},
			['minutes', 'note']
		),
		tool(
			BuiltInTool.BROWSER_RUN_SUBAGENT,
			"Hands a self-contained task to a sub-agent: a new chat with the same tools that works on the task alone and returns only its final answer to you, so its tool calls and reading do not fill your context. You wait until it is done. The sub-agent cannot see this conversation, so the task must say everything it needs and what its answer should contain. The user can open the sub-agent's chat to watch it.",
			{
				task: {
					description: 'Complete instructions for the sub-agent',
					type: JsonSchemaType.STRING
				},
				title: {
					description: 'A few words naming the task, the title of its chat',
					type: JsonSchemaType.STRING
				}
			},
			['task']
		)
	];
}
