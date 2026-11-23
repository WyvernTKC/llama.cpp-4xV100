<script lang="ts">
	// The tasks of an update_tasks list, one row each with its status icon.
	// Shared by the update_tasks block in the chat and the task strip above
	// the input box.

	import { Check, Circle, CircleDot } from '@lucide/svelte';
	import { AgentTaskStatus } from '#lib/enums/index.ts';
	import type { AgentTask } from '#lib/utils/index.ts';

	interface Props {
		tasks: readonly AgentTask[];
		class?: string;
	}

	let { class: className = '', tasks }: Props = $props();
</script>

<ul class="space-y-0.5 text-body-sm {className}">
	{#each tasks as task, i (i)}
		<li class="flex items-start gap-2">
			{#if task.status === AgentTaskStatus.COMPLETED}
				<Check class="mt-0.5 h-3.5 w-3.5 shrink-0 text-green-600 dark:text-green-400" />
			{:else if task.status === AgentTaskStatus.IN_PROGRESS}
				<CircleDot class="mt-0.5 h-3.5 w-3.5 shrink-0 text-blue-600 dark:text-blue-400" />
			{:else}
				<Circle class="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground/60" />
			{/if}

			<span
				class={task.status === AgentTaskStatus.COMPLETED
					? 'text-muted-foreground line-through'
					: task.status === AgentTaskStatus.IN_PROGRESS
						? 'font-medium text-foreground'
						: 'text-foreground/80'}
			>
				{task.content}
			</span>
		</li>
	{/each}
</ul>
