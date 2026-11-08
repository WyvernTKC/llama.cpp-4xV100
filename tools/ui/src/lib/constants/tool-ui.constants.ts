// Registry of server and browser tools whose renderer
// shows a recognizable icon and friendly label inline in the chat UI.
//
// To add a new tool, add an entry to TOOL_UI. To give a
// tool a custom title or body renderer, add a dedicated component under
// ChatMessageToolCall/ and route it in ChatMessageToolCallBlock.svelte
// (see ChatMessageToolCallBlockGetDatetime and
// ChatMessageToolCallBlockSearchResults for prior art).

import {
	AlarmClock,
	Bell,
	BookmarkPlus,
	BookmarkX,
	BookOpen,
	Bot,
	Braces,
	Brain,
	Clock,
	Eye,
	FilePen,
	FilePlus,
	FileSearch,
	FileText,
	Gauge,
	Hourglass,
	Info,
	ListChecks,
	MessageCircleQuestionMark,
	MessageSquarePlus,
	MessagesSquare,
	PenLine,
	SearchCode,
	Shrink,
	Terminal
} from '@lucide/svelte';
import { BuiltInTool, ToolSource } from '$lib/enums';
import type { ToolUiEntry } from '$lib/types';

export const TOOL_UI: Readonly<Record<BuiltInTool, ToolUiEntry>> = {
	[BuiltInTool.BROWSER_ASK_USER]: {
		icon: MessageCircleQuestionMark,
		label: 'Ask user',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_COMPACT_CONVERSATION]: {
		icon: Shrink,
		label: 'Compact conversation',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_DELETE_MEMORY]: {
		icon: BookmarkX,
		label: 'Delete memory',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_GET_CONTEXT_USAGE]: {
		icon: Gauge,
		label: 'Context usage',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_GET_DATETIME]: {
		icon: Clock,
		label: 'Current time',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_NEW_CONVERSATION]: {
		icon: MessageSquarePlus,
		label: 'New conversation',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_NOTIFY]: { icon: Bell, label: 'Notify', source: ToolSource.BROWSER },
	[BuiltInTool.BROWSER_READ_CONVERSATION]: {
		icon: BookOpen,
		label: 'Read conversation',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_READ_MEDIA]: { icon: Eye, label: 'Read media', source: ToolSource.BROWSER },
	[BuiltInTool.BROWSER_RENAME_CONVERSATION]: {
		icon: PenLine,
		label: 'Rename conversation',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_RUN_JAVASCRIPT]: {
		icon: Braces,
		label: 'Run JavaScript',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_RUN_SUBAGENT]: { icon: Bot, label: 'Sub-agent', source: ToolSource.BROWSER },
	[BuiltInTool.BROWSER_SAVE_MEMORY]: {
		icon: BookmarkPlus,
		label: 'Save memory',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_SCHEDULE_WAKEUP]: {
		icon: AlarmClock,
		label: 'Schedule wake-up',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_SEARCH_CONVERSATIONS]: {
		icon: MessagesSquare,
		label: 'Search conversations',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_SEARCH_MEMORIES]: {
		icon: Brain,
		label: 'Search memories',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_SLEEP]: { icon: Hourglass, label: 'Sleep', source: ToolSource.BROWSER },
	[BuiltInTool.BROWSER_UPDATE_TASKS]: {
		icon: ListChecks,
		label: 'Task list',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.SERVER_EDIT_FILE]: { icon: FilePen, label: 'Edit file', source: ToolSource.SERVER },
	[BuiltInTool.SERVER_EXEC_SHELL_COMMAND]: {
		icon: Terminal,
		label: 'Run command',
		source: ToolSource.SERVER
	},
	[BuiltInTool.SERVER_FILE_GLOB_SEARCH]: {
		icon: FileSearch,
		label: 'Search files',
		source: ToolSource.SERVER
	},
	[BuiltInTool.SERVER_GET_INFO]: { icon: Info, label: 'Runtime info', source: ToolSource.SERVER },
	[BuiltInTool.SERVER_GREP_SEARCH]: {
		icon: SearchCode,
		label: 'Search in files',
		source: ToolSource.SERVER
	},
	[BuiltInTool.SERVER_READ_FILE]: { icon: FileText, label: 'Read file', source: ToolSource.SERVER },
	[BuiltInTool.SERVER_WRITE_FILE]: {
		icon: FilePlus,
		label: 'Write file',
		source: ToolSource.SERVER
	}
} as const;
