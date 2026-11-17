<script lang="ts">
	// What the model keeps for the open chat above the input box: its task
	// list while some task is not done (from the newest update_tasks call on
	// the branch), and the wake-up it scheduled, which the user can cancel.

	import { AlarmClock, ChevronDown, ChevronUp, ListChecks, X } from '@lucide/svelte';
	import { AgentTaskList } from '#lib/components/app/index.ts';
	import { AgentTaskStatus } from '#lib/enums/index.ts';
	import { conversationsStore, wakeupsStore } from '#lib/stores/index.ts';
	import { latestTaskList, taskProgress } from '#lib/utils/index.ts';

	const convId = $derived(conversationsStore.activeConversation?.id ?? '');
	const tasks = $derived(convId ? latestTaskList(conversationsStore.activeMessages) : null);
	const openTasks = $derived(
		tasks && tasks.some((t) => t.status !== AgentTaskStatus.COMPLETED) ? tasks : null
	);
	const progress = $derived(openTasks ? taskProgress(openTasks) : null);
	const wakeup = $derived(convId ? wakeupsStore.get(convId) : null);
	const wakeupAt = $derived(
		wakeup
			? new Date(wakeup.dueAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
			: ''
	);

	let expanded = $state(false);
</script>

{#if openTasks || wakeup}
	<div
		class="pointer-events-auto mx-auto mb-2 flex max-w-[48rem] flex-col gap-1.5 rounded-md border border-border bg-card/95 px-3 py-1.5 text-sm shadow-sm backdrop-blur"
	>
		{#if openTasks && progress}
			<button
				class="flex w-full items-center gap-2 text-left text-muted-foreground"
				onclick={() => (expanded = !expanded)}
				type="button"
			>
				<ListChecks class="h-3.5 w-3.5 shrink-0" />

				<span class="text-xs font-medium text-foreground/80">
					Tasks {progress.done}/{progress.total}
				</span>

				{#if progress.current && !expanded}
					<span class="min-w-0 truncate text-xs">· {progress.current.content}</span>
				{/if}

				{#if expanded}
					<ChevronDown class="ml-auto h-3.5 w-3.5 shrink-0" />
				{:else}
					<ChevronUp class="ml-auto h-3.5 w-3.5 shrink-0" />
				{/if}
			</button>

			{#if expanded}
				<AgentTaskList class="max-h-48 overflow-auto pb-1 pl-5" tasks={openTasks} />
			{/if}
		{/if}

		{#if wakeup}
			<div class="flex items-center gap-2 text-xs text-muted-foreground">
				<AlarmClock class="h-3.5 w-3.5 shrink-0" />

				<span class="min-w-0 truncate">
					<span class="font-medium text-foreground/80">Wakes up at {wakeupAt}</span>
					· {wakeup.note}
				</span>

				<button
					aria-label="Cancel the wake-up"
					class="ml-auto rounded p-0.5 hover:bg-muted hover:text-foreground"
					onclick={() => wakeupsStore.cancel(convId)}
					title="Cancel the wake-up"
					type="button"
				>
					<X class="h-3.5 w-3.5" />
				</button>
			</div>
		{/if}
	</div>
{/if}
