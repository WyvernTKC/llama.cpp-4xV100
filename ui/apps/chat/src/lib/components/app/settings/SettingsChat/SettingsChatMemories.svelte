<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Trash2 } from '@lucide/svelte';
	import { DialogConfirmation } from '#lib/components/app/index.ts';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';
	import { MemoryService } from '#lib/services/memory.service.ts';
	import type { DatabaseMemory } from '#lib/types/index.ts';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	let memories = $state<DatabaseMemory[]>([]);
	let loaded = $state(false);
	let showClearDialog = $state(false);

	async function load() {
		try {
			memories = await MemoryService.list();
		} catch (error) {
			toast.error(
				`Could not read the memories: ${error instanceof Error ? error.message : String(error)}`
			);
		} finally {
			loaded = true;
		}
	}

	async function remove(id: string) {
		await MemoryService.remove(id);
		await load();
	}

	async function clearAll() {
		showClearDialog = false;
		await MemoryService.clear();
		await load();
	}

	onMount(() => {
		void load();
	});
</script>

<div class="grid gap-1">
	<h4 class="mt-0 mb-2 text-body-sm font-medium">Saved memories</h4>

	<p class="mb-4 text-body-sm text-muted-foreground">
		What the model saved with save_memory, newest first. New chats list them in their system prompt;
		a chat that already started keeps the list it started with.
	</p>

	{#if loaded && memories.length === 0}
		<p class="text-body-sm text-muted-foreground italic">No memories yet.</p>
	{:else}
		<ul class="divide-y divide-border/50 rounded-lg border border-border/50">
			{#each memories as memory (memory.id)}
				<li class="flex items-start gap-2 p-3 text-body-sm">
					<div class="min-w-0 flex-1">
						<p class="break-words">{memory.content}</p>

						<p class="mt-1 text-caption text-muted-foreground">
							{memory.id} · {new Date(memory.updatedAt).toLocaleDateString()}
						</p>
					</div>

					<Button
						aria-label="Delete memory"
						class="h-8 w-8 shrink-0 p-0"
						onclick={() => remove(memory.id)}
						variant="ghost"
					>
						<Trash2 class={ICON_CLASS_DEFAULT} />
					</Button>
				</li>
			{/each}
		</ul>

		{#if memories.length > 1}
			<Button
				class="mt-3 justify-self-start"
				onclick={() => (showClearDialog = true)}
				variant="outline"
			>
				Delete all
			</Button>
		{/if}
	{/if}
</div>

<DialogConfirmation
	cancelText="Cancel"
	confirmText="Delete all"
	description="This deletes every saved memory. New chats will start without them. This cannot be undone."
	icon={Trash2}
	onCancel={() => (showClearDialog = false)}
	onConfirm={clearAll}
	open={showClearDialog}
	title="Delete all memories"
	variant="destructive"
/>
