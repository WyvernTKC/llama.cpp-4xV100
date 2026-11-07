<script lang="ts">
	import { AudioLines, Loader2, Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';

	interface Props {
		disabled?: boolean;
		isDictating?: boolean;
		isTranscribing?: boolean;
		onDictateClick?: () => void;
	}

	let {
		disabled = false,
		isDictating = false,
		isTranscribing = false,
		onDictateClick
	}: Props = $props();

	let label = $derived(
		isTranscribing ? 'Transcribing...' : isDictating ? 'Stop dictation' : 'Dictate message'
	);
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<Button
			class="h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9 {isDictating
				? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
				: ''}"
			disabled={disabled || isTranscribing}
			onclick={onDictateClick}
			type="button"
			variant={isDictating ? 'default' : 'secondary'}
		>
			<span class="sr-only">{label}</span>

			{#if isTranscribing}
				<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin" />
			{:else if isDictating}
				<Square class="{ICON_CLASS_DEFAULT} fill-white" />
			{:else}
				<AudioLines class={ICON_CLASS_DEFAULT} />
			{/if}
		</Button>
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{isDictating || isTranscribing ? label : 'Dictate (the reply is read aloud)'}</p>
	</Tooltip.Content>
</Tooltip.Root>
