<script lang="ts">
	// Fall-through renderer for tool calls without a dedicated block.
	// Renders section.toolArgs / section.toolResult directly using the
	// shared chrome shell.

	import ToolCallBlock from './ToolCallBlock.svelte';
	import { Loader2 } from '@lucide/svelte';
	import {
		DialogChatAttachmentsPreview,
		MarkdownContent,
		SyntaxHighlightedCode
	} from '$lib/components/app';
	import { MAX_HEIGHT_CODE_BLOCK } from '$lib/constants';
	import { AttachmentType, FileTypeText, MimeTypeAudio, ToolResultKind } from '$lib/enums';
	import type {
		AgenticSection,
		DatabaseMessageExtra,
		DatabaseMessageExtraImageFile,
		ToolResultLine
	} from '$lib/types';
	import {
		classifyToolResult,
		formatJsonPretty,
		getToolUi,
		parseToolResultWithMedia
	} from '$lib/utils';
	import { createBase64DataUrl } from '$lib/utils/data-url';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let { attachments, isStreaming, onToggle, open, section }: Props = $props();

	const title = $derived(getToolUi(section.toolName)?.label ?? section.toolName ?? '');
	const outputKind = $derived(classifyToolResult(section.toolResult));
	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithMedia(section.toolResult, attachments) : []
	);

	// shown under the block, so they stay visible while it is collapsed; older
	// chats stored each tool image twice, and the name is unique per image
	const imageAttachments = $derived([
		...new Map(
			(attachments ?? [])
				.filter((a): a is DatabaseMessageExtraImageFile => a.type === AttachmentType.IMAGE)
				.map((a) => [a.name, a])
		).values()
	]);

	let previewOpen = $state(false);
	let previewIndex = $state(0);

	function openPreview(index: number) {
		previewIndex = index;
		previewOpen = true;
	}
</script>

<ToolCallBlock {isStreaming} meta={null} {onToggle} {open} {section} {title}>
	{#snippet children(_meta, ctx)}
		{#if ctx.isStreamingCall}
			<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground/70">
				<span>Input</span>

				{#if ctx.isStreaming}
					<Loader2 class="h-3 w-3 animate-spin" />
				{/if}
			</div>

			{#if section.toolArgs}
				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs)}
					language={FileTypeText.JSON}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			{:else if ctx.isStreaming}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Receiving arguments...
				</div>
			{:else}
				<div
					class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
				>
					Response was truncated
				</div>
			{/if}
		{:else}
			{@const showInput = Boolean(section.toolArgs)}
			{#if showInput}
				<div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
					<span>Input</span>
				</div>

				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs ?? '')}
					language={FileTypeText.JSON}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			{/if}

			<div
				class={showInput
					? 'mt-4 mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'
					: 'mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'}
			>
				<span>Output</span>

				{#if ctx.isPending}
					<Loader2 class="h-3 w-3 animate-spin" />
				{/if}
			</div>

			{#if ctx.isPending}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Waiting for result...
				</div>
			{:else if section.toolResult}
				{#if outputKind === ToolResultKind.JSON}
					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolResult)}
						language={FileTypeText.JSON}
						maxHeight={MAX_HEIGHT_CODE_BLOCK}
					/>
				{:else if outputKind === ToolResultKind.MARKDOWN}
					<MarkdownContent {attachments} content={section.toolResult} />
				{:else}
					<div class="overflow-auto">
						{#each parsedLines as line, i (i)}
							<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
								{line.text}
							</div>

							{#if line.media?.type === AttachmentType.AUDIO}
								{@const audioMimeType = line.media.mimeType ?? MimeTypeAudio.MP3_MPEG}
								<div class="mt-2 mb-2">
									<audio class="w-full rounded-lg" controls>
										<source
											src={createBase64DataUrl(audioMimeType, line.media.base64Data)}
											type={audioMimeType}
										/>
										Your browser does not support the audio element.
									</audio>
								</div>
							{/if}
						{/each}
					</div>
				{/if}
			{:else}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
			{/if}
		{/if}
	{/snippet}
</ToolCallBlock>

{#if imageAttachments.length > 0}
	<div class="mb-2 flex flex-col items-start gap-2">
		{#each imageAttachments as image, i (image.name)}
			<button
				aria-label="Open {image.name}"
				class="block max-w-full cursor-zoom-in rounded-lg"
				onclick={() => openPreview(i)}
				type="button"
			>
				<img
					alt={image.name}
					class="h-auto max-h-[60vh] max-w-full rounded-lg object-contain"
					loading="lazy"
					src={image.base64Url}
				/>
			</button>
		{/each}
	</div>

	<DialogChatAttachmentsPreview
		bind:open={previewOpen}
		attachments={imageAttachments}
		previewFocusIndex={previewIndex}
	/>
{/if}
