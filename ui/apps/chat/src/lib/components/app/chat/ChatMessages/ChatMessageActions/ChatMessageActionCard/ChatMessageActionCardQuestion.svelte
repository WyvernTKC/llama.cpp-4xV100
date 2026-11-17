<script lang="ts">
	import ChatMessageActionCard from './ChatMessageActionCard.svelte';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { MessageCircleQuestionMark } from '@lucide/svelte';

	interface Props {
		question: string;
		options: string[];
		onAnswer: (answer: string) => void;
	}

	let { onAnswer, options, question }: Props = $props();
</script>

<ChatMessageActionCard icon={MessageCircleQuestionMark}>
	{#snippet message()}
		{question}
	{/snippet}

	{#snippet actions()}
		{#each options as option (option)}
			<Button
				class="h-auto min-h-8 whitespace-normal"
				onclick={() => onAnswer(option)}
				size="sm"
				variant="secondary"
			>
				{option}
			</Button>
		{/each}

		<span class="text-xs text-muted-foreground">or type your own answer below</span>
	{/snippet}
</ChatMessageActionCard>
