<script lang="ts">
	// run_subagent: the task, a link to the sub-agent's own chat, and its
	// answer once it is done. While it works, a tool permission it waits for
	// shows under the block, so the user can answer it without opening the
	// sub-agent's chat.

	import ToolCallBlock from './ToolCallBlock.svelte';
	import { ExternalLink, Loader2 } from '@lucide/svelte';
	import {
		ChatMessageActionCardPermissionRequest,
		MarkdownContent
	} from '#lib/components/app/index.ts';
	import { ToolPermissionDecision } from '#lib/enums/index.ts';
	import { RouterService } from '#lib/services/index.ts';
	import { agenticStore } from '#lib/stores/index.ts';
	import type { AgenticSection } from '#lib/types/index.ts';
	import { subagentTitle } from '#lib/utils/index.ts';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	type SubagentMeta = {
		answer?: string;
		conversationId?: string;
		errorMessage?: string;
		note?: string;
		title?: string;
	};

	function parseArgs(raw: string | undefined): { task: string; title?: unknown } {
		try {
			const args: unknown = JSON.parse(raw ?? '');

			if (args && typeof args === 'object') {
				const obj = args as Record<string, unknown>;

				return { task: typeof obj.task === 'string' ? obj.task : '', title: obj.title };
			}
		} catch {
			// still streaming
		}

		return { task: '' };
	}

	function parseResult(raw: string | undefined): SubagentMeta {
		if (!raw) return {};

		try {
			const parsed: unknown = JSON.parse(raw);

			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				const obj = parsed as Record<string, unknown>;
				const text = (v: unknown) => (typeof v === 'string' && v ? v : undefined);

				return {
					answer: text(obj.answer),
					conversationId: text(obj.conversation_id),
					errorMessage: text(obj.error),
					note: text(obj.note),
					title: text(obj.title)
				};
			}
		} catch {
			return { errorMessage: raw };
		}

		return {};
	}

	const args = $derived(parseArgs(section.toolArgs));
	const meta = $derived(parseResult(section.toolResult));
	const title = $derived(meta.title ?? subagentTitle(args.title, args.task));
	const chatId = $derived(
		meta.conversationId ??
			(section.toolCallId ? agenticStore.getSubagentConversation(section.toolCallId) : null)
	);
	const working = $derived(Boolean(chatId) && !section.toolResult && isStreaming);
	const permission = $derived(
		working && chatId ? agenticStore.getPendingPermissionRequest(chatId) : null
	);

	// dismissal applies to the request object, so the next request shows again
	let dismissedPermission: typeof permission = $state(null);

	const visiblePermission = $derived(
		permission && dismissedPermission !== permission ? permission : null
	);

	function handlePermission(decision: ToolPermissionDecision) {
		if (!chatId) return;

		dismissedPermission = permission;
		agenticStore.resolvePermission(chatId, decision);
	}
</script>

<ToolCallBlock {isStreaming} {meta} {onToggle} {open} {section} {title}>
	{#snippet children(_meta, ctx)}
		{#if args.task}
			<div class="mb-1.5 text-caption text-muted-foreground/70">Task</div>

			<div class="max-h-48 overflow-auto rounded bg-muted/20 p-2 text-body-sm">
				<MarkdownContent content={args.task} />
			</div>
		{/if}

		{#if chatId}
			<a
				class="mt-2 inline-flex items-center gap-1.5 text-caption text-primary hover:underline"
				href={RouterService.chat(chatId)}
			>
				<ExternalLink class="h-3 w-3" />
				Open the sub-agent's chat
			</a>
		{/if}

		{#if working || ctx.isPending}
			<div class="mt-3 flex items-center gap-2 text-caption text-muted-foreground/70">
				<Loader2 class="h-3 w-3 animate-spin" />

				<span>{visiblePermission ? 'Waiting for your permission' : 'Working on the task...'}</span>
			</div>
		{:else if meta.answer}
			<div class="mt-4 mb-1.5 text-caption text-muted-foreground/70">Answer</div>

			<MarkdownContent content={meta.answer} />
		{/if}

		{#if meta.note || meta.errorMessage}
			<div class="mt-2 text-caption text-muted-foreground italic">
				{meta.note ?? meta.errorMessage}
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>

{#if visiblePermission}
	<ChatMessageActionCardPermissionRequest
		onDecision={handlePermission}
		serverLabel={visiblePermission.serverLabel}
		toolName={visiblePermission.toolName}
	/>
{/if}
