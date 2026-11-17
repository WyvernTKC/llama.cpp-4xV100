import {
	AGENT_TOOL_NAMES,
	AGENT_TOOLS,
	buildAgentToolDefinitions,
	SUBAGENT_DISABLED_TOOLS,
	TOOL_UI
} from '#lib/constants/index.ts';
import { AgentTaskStatus, BuiltInTool, MessageRole, MessageType } from '#lib/enums/index.ts';
import { AgenticGates } from '#lib/stores/agentic/gates.svelte.ts';
import type { DatabaseMessage } from '#lib/types/index.ts';
import {
	clampToolNumber,
	describeTaskList,
	latestTaskList,
	normalizeAskArgs,
	normalizeNotifyArgs,
	normalizeTaskList,
	subagentPrompt,
	subagentTitle,
	taskProgress,
	truncateSubagentAnswer,
	wakeupMessage
} from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

function assistantWithCalls(
	id: string,
	calls: { name: string; arguments: string }[]
): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'conv',
		id,
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: JSON.stringify(
			calls.map((fn, i) => ({ function: fn, id: `${id}-${i}`, type: 'function' }))
		),
		type: MessageType.TEXT
	} as DatabaseMessage;
}

function tasksArgs(...tasks: [string, string][]): string {
	return JSON.stringify({ tasks: tasks.map(([content, status]) => ({ content, status })) });
}

describe('agent tool definitions', () => {
	it('covers every agent tool name once, each with a tool-call label', () => {
		const names = buildAgentToolDefinitions().map((t) => t.function.name);

		expect([...names].sort()).toEqual([...AGENT_TOOL_NAMES].sort());

		for (const name of names) expect(TOOL_UI[name as BuiltInTool]).toBeDefined();
	});

	it('keeps run_subagent and the waiting tools away from sub-agents', () => {
		expect(SUBAGENT_DISABLED_TOOLS).toContain(BuiltInTool.BROWSER_RUN_SUBAGENT);
		expect(SUBAGENT_DISABLED_TOOLS).toContain(BuiltInTool.BROWSER_ASK_USER);
		expect(SUBAGENT_DISABLED_TOOLS).not.toContain(BuiltInTool.BROWSER_UPDATE_TASKS);
	});

	it('declares update_tasks statuses as an enum on array items', () => {
		const def = buildAgentToolDefinitions().find(
			(t) => t.function.name === BuiltInTool.BROWSER_UPDATE_TASKS
		);
		const tasks = (def?.function.parameters as { properties: Record<string, never> }).properties
			.tasks as { type: string; items: { properties: { status: { enum: string[] } } } };

		expect(tasks.type).toBe('array');
		expect(tasks.items.properties.status.enum).toEqual(['pending', 'in_progress', 'completed']);
	});
});

describe('normalizeTaskList', () => {
	it('reads the list from the arguments, a bare array or a JSON string', () => {
		const list = [{ content: 'a', status: 'completed' }];

		expect(normalizeTaskList({ tasks: list })).toEqual([
			{ content: 'a', status: AgentTaskStatus.COMPLETED }
		]);
		expect(normalizeTaskList(list)).toHaveLength(1);
		expect(normalizeTaskList({ tasks: JSON.stringify(list) })).toHaveLength(1);
	});

	it('drops tasks without text and reads an unknown status as pending', () => {
		expect(
			normalizeTaskList({
				tasks: [{ content: '  ' }, { content: ' do  it ', status: 'started' }, 'text', null]
			})
		).toEqual([{ content: 'do it', status: AgentTaskStatus.PENDING }]);
	});

	it('is null without a list and caps a long one', () => {
		expect(normalizeTaskList({ tasks: 'nope' })).toBeNull();
		expect(normalizeTaskList({})).toBeNull();

		const many = Array.from({ length: AGENT_TOOLS.TASKS_MAX + 5 }, (_, i) => ({
			content: `t${i}`,
			status: 'pending'
		}));

		expect(normalizeTaskList({ tasks: many })).toHaveLength(AGENT_TOOLS.TASKS_MAX);
	});

	it('describes the progress', () => {
		const tasks = normalizeTaskList({
			tasks: [
				{ content: 'a', status: 'completed' },
				{ content: 'b', status: 'in_progress' },
				{ content: 'c', status: 'pending' }
			]
		})!;

		expect(taskProgress(tasks)).toEqual({ current: tasks[1], done: 1, total: 3 });
		expect(describeTaskList(tasks)).toBe('Task list updated: 1 of 3 done. In progress: b.');
		expect(describeTaskList([])).toBe('Task list cleared.');
	});
});

describe('latestTaskList', () => {
	it('takes the last update_tasks call on the branch', () => {
		const messages = [
			assistantWithCalls('m1', [
				{ arguments: tasksArgs(['old', 'pending']), name: BuiltInTool.BROWSER_UPDATE_TASKS }
			]),
			assistantWithCalls('m2', [
				{ arguments: '{}', name: BuiltInTool.BROWSER_GET_DATETIME },
				{ arguments: tasksArgs(['new', 'in_progress']), name: BuiltInTool.BROWSER_UPDATE_TASKS }
			])
		];

		expect(latestTaskList(messages)).toEqual([
			{ content: 'new', status: AgentTaskStatus.IN_PROGRESS }
		]);
	});

	it('keeps the previous list while a call is still streaming', () => {
		const messages = [
			assistantWithCalls('m1', [
				{ arguments: tasksArgs(['done', 'completed']), name: BuiltInTool.BROWSER_UPDATE_TASKS }
			]),
			assistantWithCalls('m2', [
				{ arguments: '{"tasks":[{"content":"ha', name: BuiltInTool.BROWSER_UPDATE_TASKS }
			])
		];

		expect(latestTaskList(messages)?.[0].content).toBe('done');
	});

	it('is null on a branch without one', () => {
		expect(
			latestTaskList([
				assistantWithCalls('m1', [{ arguments: '{}', name: BuiltInTool.BROWSER_GET_DATETIME }])
			])
		).toBeNull();
	});
});

describe('normalizeAskArgs', () => {
	it('cleans, dedupes and caps the options', () => {
		const ask = normalizeAskArgs({
			options: ['Yes', ' yes ', 'Yes', 'No', '', 3, 'a', 'b', 'c', 'd', 'e'],
			question: ' Go on? '
		});

		expect(ask?.question).toBe('Go on?');
		expect(ask?.options).toEqual(['Yes', 'yes', 'No', '3', 'a', 'b']);
		expect(ask?.options).toHaveLength(AGENT_TOOLS.ASK_MAX_OPTIONS);
	});

	it('accepts options as a JSON string or lines', () => {
		expect(normalizeAskArgs({ options: '["A","B"]', question: 'q' })?.options).toEqual(['A', 'B']);
		expect(normalizeAskArgs({ options: 'A\nB', question: 'q' })?.options).toEqual(['A', 'B']);
	});

	it('needs a question and two options', () => {
		expect(normalizeAskArgs({ options: ['A', 'B'], question: '' })).toBeNull();
		expect(normalizeAskArgs({ options: ['A', 'A'], question: 'q' })).toBeNull();
	});
});

describe('small agent tool helpers', () => {
	it('clamps numbers and reads junk as 0', () => {
		expect(clampToolNumber(30, 60)).toBe(30);
		expect(clampToolNumber('90', 60)).toBe(60);
		expect(clampToolNumber(-5, 60)).toBe(0);
		expect(clampToolNumber('soon', 60)).toBe(0);
		expect(clampToolNumber(undefined, 60)).toBe(0);
	});

	it('builds the wake-up message from the note', () => {
		expect(wakeupMessage(' check the\n build ')).toBe(
			`${AGENT_TOOLS.WAKEUP_PREFIX} check the build`
		);
	});

	it('needs notification text and defaults the title', () => {
		expect(normalizeNotifyArgs({ message: '  ' })).toBeNull();
		expect(normalizeNotifyArgs({ message: 'Done' })).toEqual({
			message: 'Done',
			title: AGENT_TOOLS.NOTIFY_DEFAULT_TITLE
		});
		expect(normalizeNotifyArgs({ message: 'Done', title: 'Build' })?.title).toBe('Build');
	});

	it('titles a sub-agent chat by its title or its task', () => {
		expect(subagentTitle('Find logs', 'x')).toBe(`${AGENT_TOOLS.SUBAGENT_TITLE_PREFIX}Find logs`);
		expect(subagentTitle(undefined, 'Read every file in the src folder and list the exports')).toBe(
			`${AGENT_TOOLS.SUBAGENT_TITLE_PREFIX}Read every file in the src`
		);
	});

	it('puts the task last in the sub-agent prompt', () => {
		expect(subagentPrompt('  do X  ').endsWith('Task:\ndo X')).toBe(true);
	});

	it('cuts a long answer with a note', () => {
		const long = 'a'.repeat(AGENT_TOOLS.SUBAGENT_ANSWER_MAX_CHARS + 10);
		const cut = truncateSubagentAnswer(long);

		expect(cut.startsWith('a'.repeat(AGENT_TOOLS.SUBAGENT_ANSWER_MAX_CHARS))).toBe(true);
		expect(cut).toContain('10 more characters');
		expect(truncateSubagentAnswer('short')).toBe('short');
	});
});

describe('ask_user gate', () => {
	const question = { options: ['A', 'B'], question: 'q', toolCallId: 't1' };

	it('resolves with the answer and clears the question', async () => {
		const gates = new AgenticGates();
		const pending = gates.requestAnswer('c', question);

		expect(gates.getPendingQuestion('c')).toEqual(question);
		expect(gates.resolveQuestion('c', 'A')).toBe(true);
		await expect(pending).resolves.toBe('A');
		expect(gates.getPendingQuestion('c')).toBeNull();
		expect(gates.resolveQuestion('c', 'B')).toBe(false);
	});

	it('resolves null on a stop or when the flow state is cleared', async () => {
		const gates = new AgenticGates();
		const controller = new AbortController();
		const stopped = gates.requestAnswer('c', question, controller.signal);

		controller.abort();
		await expect(stopped).resolves.toBeNull();
		expect(gates.getPendingQuestion('c')).toBeNull();

		const cleared = gates.requestAnswer('d', question);

		gates.clear('d');
		await expect(cleared).resolves.toBeNull();
	});
});
