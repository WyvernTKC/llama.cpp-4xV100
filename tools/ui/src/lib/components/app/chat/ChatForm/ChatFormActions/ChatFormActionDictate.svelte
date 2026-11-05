<script lang="ts">
	import { AudioLines, Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';

	interface Props {
		disabled?: boolean;
		isDictating?: boolean;
		onDictateClick?: () => void;
	}

	let { disabled = false, isDictating = false, onDictateClick }: Props = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<Button
			class="h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9 {isDictating
				? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
				: ''}"
			{disabled}
			onclick={onDictateClick}
			type="button"
			variant={isDictating ? 'default' : 'secondary'}
		>
			<span class="sr-only">{isDictating ? 'Stop dictation' : 'Dictate message'}</span>

			{#if isDictating}
				<Square class="{ICON_CLASS_DEFAULT} fill-white" />
			{:else}
				<AudioLines class={ICON_CLASS_DEFAULT} />
			{/if}
		</Button>
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{isDictating ? 'Stop dictation' : 'Dictate (the reply is read aloud)'}</p>
	</Tooltip.Content>
</Tooltip.Root>
