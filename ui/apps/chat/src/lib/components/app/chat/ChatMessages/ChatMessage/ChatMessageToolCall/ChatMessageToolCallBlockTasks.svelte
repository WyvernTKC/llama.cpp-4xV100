<script lang="ts">
	// update_tasks: one line with the progress of the list, the list itself
	// on click. The arguments are the list, so a call that is still streaming
	// shows a spinner until they parse.

	import AgentTaskList from './AgentTaskList.svelte';
	import { ChevronDown, ChevronUp, ListChecks, Loader2 } from '@lucide/svelte';
	import { AgenticSectionType } from '#lib/enums/index.ts';
	import type { AgenticSection } from '#lib/types/index.ts';
	import { normalizeTaskList, taskProgress } from '#lib/utils/index.ts';

	interface Props {
		section: AgenticSection;
		isStreaming?: boolean;
		open: boolean;
		onToggle?: () => void;
	}

	let { isStreaming = false, onToggle, open, section }: Props = $props();

	const showSpinner = $derived(
		section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			(section.type === AgenticSectionType.TOOL_CALL_STREAMING && isStreaming)
	);

	const tasks = $derived.by(() => {
		try {
			return normalizeTaskList(JSON.parse(section.toolArgs ?? ''));
		} catch {
			return null;
		}
	});

	const progress = $derived(tasks ? taskProgress(tasks) : null);
</script>

<div class="py-1.5">
	<button
		class="flex w-full items-center gap-2 text-left text-muted-foreground"
		disabled={!tasks || tasks.length === 0}
		onclick={() => onToggle?.()}
		type="button"
	>
		<ListChecks class="h-3.5 w-3.5 shrink-0 text-muted-foreground/60" />

		<span class="text-body-sm font-medium text-foreground/80">Task list</span>

		{#if progress && progress.total > 0}
			<span class="text-caption">{progress.done}/{progress.total} done</span>

			{#if progress.current}
				<span class="min-w-0 truncate text-caption">· {progress.current.content}</span>
			{/if}
		{:else if tasks}
			<span class="text-caption">cleared</span>
		{/if}

		{#if showSpinner}
			<Loader2 class="h-3 w-3 animate-spin text-muted-foreground/70" />
		{:else if tasks && tasks.length > 0}
			{#if open}
				<ChevronUp class="ml-auto h-3.5 w-3.5 shrink-0" />
			{:else}
				<ChevronDown class="ml-auto h-3.5 w-3.5 shrink-0" />
			{/if}
		{/if}
	</button>

	{#if open && tasks && tasks.length > 0}
		<AgentTaskList class="mt-1.5 ml-5" {tasks} />
	{/if}
</div>
