import { normalizeConversationTitle } from './chat-tools';
import { AGENT_TOOLS } from '$lib/constants';
import { AgentTaskStatus, BuiltInTool, MessageRole } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';

/** One entry of the task list the model keeps with update_tasks. */
export interface AgentTask {
	content: string;
	status: AgentTaskStatus;
}

/** How far a task list has come. */
export interface AgentTaskProgress {
	current: AgentTask | null;
	done: number;
	total: number;
}

const TASK_STATUSES: readonly string[] = Object.values(AgentTaskStatus);

function collapseWhitespace(text: string): string {
	return text.replace(/\s+/g, ' ').trim();
}

/** A list argument as the model sent it: an array, or an array encoded as a JSON string. */
function asArray(value: unknown): unknown[] | null {
	if (Array.isArray(value)) return value;

	if (typeof value !== 'string') return null;

	try {
		const parsed = JSON.parse(value);

		return Array.isArray(parsed) ? parsed : null;
	} catch {
		return null;
	}
}

/**
 * The task list in update_tasks arguments, cleaned up: tasks without text are
 * dropped, an unknown status reads as pending, and the list is capped. Null
 * when the arguments hold no list at all.
 */
export function normalizeTaskList(args: unknown): AgentTask[] | null {
	const raw = args && typeof args === 'object' && 'tasks' in args ? args.tasks : args;
	const list = asArray(raw);

	if (!list) return null;

	const tasks: AgentTask[] = [];

	for (const item of list) {
		if (tasks.length >= AGENT_TOOLS.TASKS_MAX) break;

		if (!item || typeof item !== 'object') continue;

		const entry = item as { content?: unknown; status?: unknown };
		const content =
			typeof entry.content === 'string'
				? collapseWhitespace(entry.content).slice(0, AGENT_TOOLS.TASK_MAX_LENGTH)
				: '';

		if (!content) continue;

		const status =
			typeof entry.status === 'string' && TASK_STATUSES.includes(entry.status)
				? (entry.status as AgentTaskStatus)
				: AgentTaskStatus.PENDING;

		tasks.push({ content, status });
	}

	return tasks;
}

export function taskProgress(tasks: readonly AgentTask[]): AgentTaskProgress {
	return {
		current: tasks.find((t) => t.status === AgentTaskStatus.IN_PROGRESS) ?? null,
		done: tasks.filter((t) => t.status === AgentTaskStatus.COMPLETED).length,
		total: tasks.length
	};
}

/** What update_tasks tells the model after it stored a list. */
export function describeTaskList(tasks: readonly AgentTask[]): string {
	if (tasks.length === 0) return 'Task list cleared.';

	const { current, done, total } = taskProgress(tasks);
	const now = current ? ` In progress: ${current.content}.` : '';

	return `Task list updated: ${done} of ${total} done.${now}`;
}

/**
 * The newest task list on a branch: the arguments of its last complete
 * update_tasks call. A call still streaming does not parse yet, so the list
 * before it stays until its arguments are whole.
 */
export function latestTaskList(messages: readonly DatabaseMessage[]): AgentTask[] | null {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];

		if (message.role !== MessageRole.ASSISTANT || !message.toolCalls) continue;

		let calls: unknown;

		try {
			calls = JSON.parse(message.toolCalls);
		} catch {
			continue;
		}

		if (!Array.isArray(calls)) continue;

		for (let j = calls.length - 1; j >= 0; j--) {
			const fn = (calls[j] as { function?: { arguments?: unknown; name?: unknown } })?.function;

			if (fn?.name !== BuiltInTool.BROWSER_UPDATE_TASKS) continue;

			let args: unknown = fn.arguments;

			if (typeof args === 'string') {
				try {
					args = JSON.parse(args);
				} catch {
					continue;
				}
			}

			const tasks = normalizeTaskList(args);

			if (tasks) return tasks;
		}
	}

	return null;
}

/** The question and options of ask_user arguments, cleaned up; null without a question or two options. */
export function normalizeAskArgs(
	args: Record<string, unknown>
): { options: string[]; question: string } | null {
	const question = typeof args.question === 'string' ? args.question.trim() : '';
	const raw =
		asArray(args.options) ??
		(typeof args.options === 'string' ? args.options.split(/\r?\n|\|/) : []);
	const options: string[] = [];

	for (const item of raw) {
		if (options.length >= AGENT_TOOLS.ASK_MAX_OPTIONS) break;

		if (typeof item !== 'string' && typeof item !== 'number') continue;

		const option = collapseWhitespace(String(item)).slice(0, AGENT_TOOLS.ASK_OPTION_MAX_LENGTH);

		if (option && !options.includes(option)) options.push(option);
	}

	if (!question || options.length < 2) return null;

	return { options, question };
}

/** A number argument clamped to 0..max; anything that is not a number reads as 0. */
export function clampToolNumber(value: unknown, max: number): number {
	const n = typeof value === 'string' ? Number(value.trim()) : Number(value);

	if (!Number.isFinite(n)) return 0;

	return Math.min(max, Math.max(0, n));
}

/** The message a wake-up sends to the model. */
export function wakeupMessage(note: string): string {
	return `${AGENT_TOOLS.WAKEUP_PREFIX} ${collapseWhitespace(note)}`;
}

/** The text and title of notify arguments, cleaned up; null without text. */
export function normalizeNotifyArgs(
	args: Record<string, unknown>
): { message: string; title: string } | null {
	const message =
		typeof args.message === 'string'
			? collapseWhitespace(args.message).slice(0, AGENT_TOOLS.NOTIFY_MAX_LENGTH)
			: '';

	if (!message) return null;

	const title =
		typeof args.title === 'string' && collapseWhitespace(args.title)
			? collapseWhitespace(args.title).slice(0, AGENT_TOOLS.NOTIFY_MAX_LENGTH)
			: AGENT_TOOLS.NOTIFY_DEFAULT_TITLE;

	return { message, title };
}

/** The title of a sub-agent's chat: the given title, else the first words of its task. */
export function subagentTitle(title: unknown, task: string): string {
	const named = normalizeConversationTitle(title);
	const fallback = collapseWhitespace(task).split(' ').slice(0, 6).join(' ');

	return `${AGENT_TOOLS.SUBAGENT_TITLE_PREFIX}${named || fallback || 'task'}`;
}

/** The first message of a sub-agent's chat: what it is, then the task. */
export function subagentPrompt(task: string): string {
	return [
		'You are a sub-agent. Another conversation handed you the task below and receives only your final answer, as the result of its tool call.',
		'Work on the task on your own with your tools; nobody will answer questions while you work.',
		'When you are done, reply with the final answer alone: complete and self-contained, with the facts, file paths or results the task asks for.',
		'',
		'Task:',
		task.trim()
	].join('\n');
}

/** A sub-agent's answer as the caller reads it, cut at the limit with a note where it ends. */
export function truncateSubagentAnswer(answer: string): string {
	const max = AGENT_TOOLS.SUBAGENT_ANSWER_MAX_CHARS;

	if (answer.length <= max) return answer;

	return `${answer.slice(0, max)}\n[... ${answer.length - max} more characters; the full answer is in the sub-agent's chat]`;
}
