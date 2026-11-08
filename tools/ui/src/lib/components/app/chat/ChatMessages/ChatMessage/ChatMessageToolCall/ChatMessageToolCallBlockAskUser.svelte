<script lang="ts">
	// ask_user: the question, then the answer once the user gave one. The
	// options to pick from show in a card under the reply while it waits.

	import { Loader2, MessageCircleQuestionMark } from '@lucide/svelte';
	import { AgenticSectionType } from '$lib/enums';
	import type { AgenticSection } from '$lib/types';

	interface Props {
		section: AgenticSection;
		isStreaming?: boolean;
	}

	let { isStreaming = false, section }: Props = $props();

	const waiting = $derived(
		section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			(section.type === AgenticSectionType.TOOL_CALL_STREAMING && isStreaming)
	);

	const question = $derived.by(() => {
		try {
			const args: unknown = JSON.parse(section.toolArgs ?? '');

			return args && typeof args === 'object' && 'question' in args
				? String((args as { question: unknown }).question ?? '')
				: '';
		} catch {
			return '';
		}
	});

	const answer = $derived.by(() => {
		if (!section.toolResult) return null;

		try {
			const parsed: unknown = JSON.parse(section.toolResult);

			if (parsed && typeof parsed === 'object' && 'answer' in parsed) {
				return String((parsed as { answer: unknown }).answer ?? '');
			}
		} catch {
			// a plain-text result says why there is no answer
		}

		return null;
	});
</script>

<div class="flex flex-col gap-1 py-1.5 text-muted-foreground">
	<div class="flex items-start gap-2">
		<MessageCircleQuestionMark class="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground/60" />

		<span class="text-sm text-foreground/80">
			<span class="font-medium">Asked you:</span>
			{question || '...'}
		</span>

		{#if waiting}
			<Loader2 class="mt-1 h-3 w-3 shrink-0 animate-spin text-muted-foreground/70" />
		{/if}
	</div>

	{#if answer !== null}
		<div class="ml-5 text-sm">
			<span class="font-medium text-foreground/80">You answered:</span>

			<span class="text-foreground/90">{answer}</span>
		</div>
	{:else if section.toolResult}
		<div class="ml-5 text-xs italic">{section.toolResult}</div>
	{/if}
</div>
