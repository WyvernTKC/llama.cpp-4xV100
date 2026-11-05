<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatForm } from '$lib/components/app';
	import { MessageRole } from '$lib/enums';
	import { useDraftMessages } from '$lib/hooks/use-draft-messages.svelte';
	import { conversationsStore, deviceStore, speechStore } from '$lib/stores';
	import { getAgenticText } from '$lib/utils';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		onFileRemove?: (fileId: string) => void;
		onFileUpload?: (files: File[]) => void;
		onSend?: (message: string, files?: ChatUploadedFile[]) => Promise<boolean>;
		onStop?: () => void;
		onSystemPromptAdd?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		class: className,
		disabled = false,
		isLoading = false,
		onFileRemove,
		onFileUpload,
		onSend,
		onStop,
		onSystemPromptAdd,
		uploadedFiles = $bindable([])
	}: Props = $props();

	let chatFormRef: ChatForm | undefined = $state(undefined);

	let formWrapperEl: HTMLDivElement | undefined = $state();
	let chatId = $derived(page.params.id as string | undefined);

	$effect(() => {
		if (!formWrapperEl) return;

		const formEl = formWrapperEl.querySelector('form') as HTMLElement | null;

		if (!formEl) return;

		const updateHeight = () => {
			const height = Math.round(formEl.getBoundingClientRect().height);

			document.documentElement.style.setProperty('--chat-form-height', `${height}px`);
		};

		updateHeight();

		const resizeObserver = new ResizeObserver(updateHeight);

		resizeObserver.observe(formEl);

		return () => {
			resizeObserver.disconnect();
			document.documentElement.style.removeProperty('--chat-form-height');
		};
	});
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let message = $state('');
	let previousIsLoading = $derived(isLoading);
	// send time of the dictated message whose reply is read aloud, 0 when none
	let speakReplySentAt = 0;

	const { clearDraft } = useDraftMessages({
		getChatId: () => chatId,
		getFiles: () => uploadedFiles,
		getMessage: () => message,
		setFiles: (f) => (uploadedFiles = f),
		setMessage: (m) => (message = m)
	});

	function handleFilesAdd(files: File[]) {
		onFileUpload?.(files);
	}

	async function handleSubmit() {
		if ((!message.trim() && uploadedFiles.length === 0) || disabled || hasLoadingAttachments)
			return;

		if (!chatFormRef?.checkModelSelected()) return;

		const messageToSend = message.trim();
		const filesToSend = [...uploadedFiles];
		const sentAt = Date.now();

		if (chatFormRef?.takeDictated()) speakReplySentAt = sentAt;

		message = '';
		uploadedFiles = [];
		clearDraft();

		chatFormRef?.resetTextareaHeight();

		const success = await onSend?.(messageToSend, filesToSend);

		if (!success) {
			message = messageToSend;
			uploadedFiles = filesToSend;

			return;
		}

		if (speakReplySentAt === sentAt) {
			speakReplySentAt = 0;
			speakReply(sentAt);
		}
	}

	// The send resolves once the reply is done. Read it only if the active
	// chat still ends with the message sent at sentAt (the user may have left).
	function speakReply(sentAt: number) {
		const messages = conversationsStore.activeMessages as DatabaseMessage[];

		let userIdx = messages.length - 1;

		while (userIdx >= 0 && messages[userIdx].role !== MessageRole.USER) userIdx--;

		if (userIdx < 0 || messages[userIdx].timestamp < sentAt) return;

		const [reply, ...rest] = messages.slice(userIdx + 1);

		if (reply?.role !== MessageRole.ASSISTANT) return;

		speechStore.speak(reply.id, getAgenticText(reply, rest));
	}

	function handleStop() {
		speakReplySentAt = 0;
		onStop?.();
	}

	function handleSystemPromptClick() {
		onSystemPromptAdd?.({ files: uploadedFiles, message });
	}

	function handleUploadedFileRemove(fileId: string) {
		onFileRemove?.(fileId);
	}

	// Auto-focus must not steal focus already claimed elsewhere (e.g. the system
	// message editor opened just before a navigation)
	function focusFormUnlessCaptured() {
		const active = document.activeElement;

		if (active instanceof HTMLTextAreaElement || active instanceof HTMLInputElement) return;

		chatFormRef?.focus();
	}

	onMount(() => {
		if (!deviceStore.isMobile) {
			setTimeout(focusFormUnlessCaptured, 100);
		}
	});

	afterNavigate((navigation) => {
		if (navigation?.from != null && !deviceStore.isMobile) {
			setTimeout(focusFormUnlessCaptured, 100);
		}
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			setTimeout(focusFormUnlessCaptured, 10);
		}

		previousIsLoading = isLoading;
	});
</script>

<div bind:this={formWrapperEl} class="chat-screen-form-wrapper">
	<ChatForm
		bind:this={chatFormRef}
		bind:uploadedFiles
		bind:value={message}
		class="mx-auto max-w-3xl {className}"
		{disabled}
		{isLoading}
		onFilesAdd={handleFilesAdd}
		onStop={handleStop}
		onSubmit={handleSubmit}
		onSystemPromptClick={handleSystemPromptClick}
		onUploadedFileRemove={handleUploadedFileRemove}
	/>
</div>
