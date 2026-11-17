<script lang="ts">
	import { Download } from '@lucide/svelte';

	interface Props {
		displayName: string;
		fileSize: string;
		/** Shows a Save button when set */
		onDownload?: () => void;
	}

	let { displayName, fileSize, onDownload }: Props = $props();
</script>

<div class="sticky top-0 z-[20] mb-4 rounded-lg bg-black/5 px-4 py-2 text-center backdrop-blur-md">
	<p class="font-medium text-white">{displayName}</p>

	{#if fileSize}
		<p class="text-xs text-white/60">{fileSize}</p>
	{/if}

	{#if onDownload}
		<button
			class="mx-auto mt-1 flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1 text-xs text-white/80 hover:bg-white/10 hover:text-white"
			onclick={onDownload}
			type="button"
		>
			<Download class="size-3.5" />
			Save
		</button>
	{/if}
</div>
