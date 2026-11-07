<script lang="ts">
	import ContextGaugePopup from './ChatFormContextGauge/ContextGaugePopup.svelte';
	import {
		ChatAttachmentsList,
		ChatFormActions,
		ChatFormCurrentWorkingDirectory,
		ChatFormInput,
		ChatFormInputFileInputInvisible,
		ChatFormMcpResourcesList,
		ChatFormPickers,
		DialogMcpResourcesBrowser
	} from '$lib/components/app';
	import {
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		INITIAL_FILE_SIZE,
		INPUT_CLASSES,
		PROMPT_CONTENT_SEPARATOR,
		SETTING_CONFIG_DEFAULT,
		SPEECH_MIC_BLOCKED_MESSAGE,
		SPEECH_RECOGNITION_BLOCKED_ERRORS,
		SPEECH_RECOGNITION_SILENT_ERRORS,
		TRANSCRIPTION_SAMPLE_RATE
	} from '$lib/constants';
	import {
		ContentPartType,
		FileExtensionText,
		KeyboardKey,
		MimeTypeText,
		SpecialFileType,
		ToolSource
	} from '$lib/enums';
	import { useChatFormPickers } from '$lib/hooks/use-chat-form-pickers.svelte';
	import { TranscriptionService } from '$lib/services';
	import {
		chatStore,
		conversationsStore,
		mcpStore,
		modelsStore,
		serverStore,
		settingsStore,
		speechStore,
		toolsStore,
		uiStore
	} from '$lib/stores';
	import type {
		FileMentionEntry,
		GetPromptResult,
		MCPPromptInfo,
		MCPResourceInfo,
		PromptMessage
	} from '$lib/types';
	import {
		autoResizeTextarea,
		buildMentionInsertion,
		containsCodeSpan,
		containsFileMentionLink,
		createSpeechRecognition,
		findCommandToken,
		findMentionToken,
		getConversationModel,
		isIMEComposing,
		isOffsetInCodeBlock,
		isSpeechRecognitionSupported,
		parseClipboardContent,
		type SpeechRecognitionLike,
		speechResultsToText,
		uuid
	} from '$lib/utils';
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/browser-only';
	import { onMount, tick } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		// Data
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		// UI State
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		placeholder?: string;
		showAddButton?: boolean;
		showModelSelector?: boolean;

		// Event Handlers
		onAttachmentRemove?: (index: number) => void;
		onFilesAdd?: (files: File[]) => void;
		onStop?: () => void;
		onSubmit?: () => void;
		onSystemPromptClick?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		onUploadedFileRemove?: (fileId: string) => void;
		onUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
		onValueChange?: (value: string) => void;
	}

	let {
		attachments = [],
		class: className = '',
		disabled = false,
		isLoading = false,
		onAttachmentRemove,
		onFilesAdd,
		onStop,
		onSubmit,
		onSystemPromptClick,
		onUploadedFileRemove,
		onUploadedFilesChange,
		onValueChange,
		placeholder = 'Type a message...',
		showAddButton = true,
		showModelSelector = true,
		uploadedFiles = $bindable([]),
		value = $bindable('')
	}: Props = $props();

	// Component References
	// Shared handle of the two input renderers (plain textarea + rich chat form input).
	type ChatInputHandle = {
		focus(): void;
		resetHeight(): void;
		getElement(): HTMLElement | undefined;
		getCaretOffset(): number;
		setCaretOffset(offset: number): void;
	};

	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormInputFileInputInvisible | undefined = $state(undefined);
	let pickersRef: { handleKeydown: (event: KeyboardEvent) => boolean } | undefined =
		$state(undefined);
	let inputRef: ChatInputHandle | undefined = $state(undefined);

	// Render-mode gate: the plain textarea by default, the rich chat form input
	// while the buffer carries a `file://` mention link or a complete code
	// span (badges and code chips need a DOM the textarea cannot provide).
	// Demotes back once neither remains.
	let useRichInput = $state(false);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	// Dictation State: `dictated` marks input that came from speech, so the reply is read aloud
	let recognition: SpeechRecognitionLike | null = null;
	let dictationRecorder: AudioRecorder | undefined;
	let isDictating = $state(false);
	let isTranscribing = $state(false);
	let speechRecognitionSupported = $state(false);
	let mediaRecorderSupported = $state(false);
	let dictated = false;
	// bumped when the input is sent, so a transcript that comes back later is dropped
	let dictationRun = 0;

	// Invisible anchor at the form's top edge so the mention/WD popovers
	// float above the box.
	let mentionAnchor: HTMLDivElement | null = $state(null);

	let cwd = $derived(
		conversationsStore.activeConversation?.cwd ?? conversationsStore.preferences.pendingCwd
	);

	const pickers = useChatFormPickers({
		canCompact: () => chatStore.canCompact(conversationsStore.activeMessages),
		focusInput: refocusInput,
		getCaretOffset: () => inputRef?.getCaretOffset(),
		getCwd: () => cwd,
		getPickersRef: () => pickersRef,
		getServerHome: () => toolsStore.serverHome ?? null,
		getShowModelSelector: () => showModelSelector,
		getValue: () => value,
		hasCwdTools: () => conversationsStore.preferences.hasEnabledCwdTools(),
		// policy-aware, same rule as the agentic flow: MCP category on and at
		// least one globally-enabled server whose group key is not disabled
		hasPrompts: () => {
			const prefs = conversationsStore.preferences;

			if (!prefs.isCategoryEnabled(ToolSource.MCP)) return false;

			return mcpStore
				.getServers()
				.some((s) => s.enabled && prefs.isServerToolsEnabled(s.id) && s.url.trim());
		},
		onCompact: () => chatStore.compactConversation(),
		openModelSelector: () => chatFormActionsRef?.openModelSelector(),
		setCaretOffset: (offset) => inputRef?.setCaretOffset(offset),
		setValue: (v) => {
			value = v;
			onValueChange?.(v);
		}
	});

	async function handleWorkingDirectoryChange(newDir: string | null) {
		// Committing a directory consumes the `/cwd` token; the chip's
		// clear-X path has no token to consume.
		const token = findCommandToken(value);

		if (token && token.name === 'cwd') {
			value = '';
			onValueChange?.('');
		}

		await conversationsStore.preferences.setCwd(newDir);

		if (conversationsStore.activeConversation) {
			await chatStore.recordCwdChange(newDir?.trim() || null);
		}
	}

	// Resource Dialog State
	let isResourceDialogOpen = $state(false);
	let preSelectedResourceUri = $state<string | undefined>(undefined);

	// MCP Servers Dialog State

	let currentConfig = $derived(settingsStore.config);

	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);

		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

	let isRouter = $derived(serverStore.isRouterMode);
	let useServerDictation = $derived(Boolean(currentConfig.dictationUseServer));
	let conversationModel = $derived(
		getConversationModel(conversationsStore.activeMessages as DatabaseMessage[])
	);
	let activeModelId = $derived(modelsStore.activeModelId);

	let hasModelSelected = $derived(
		!isRouter || !!conversationModel || !!modelsStore.selectedModelId
	);
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let hasAttachments = $derived(
		(attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
	);
	let canSubmit = $derived(value.trim().length > 0 || hasAttachments);

	// Caret offset restored after a renderer swap. Callers that mutate
	// `value` themselves (e.g. the mention picker) pin the target offset
	// BEFORE the assignment; otherwise the swap effect snapshots the
	// current caret.
	let pendingCaretOffset = 0;
	let caretOffsetPinned = false;

	function queueCaretRestore() {
		queueMicrotask(() => {
			inputRef?.focus();
			inputRef?.setCaretOffset(pendingCaretOffset);
			caretOffsetPinned = false;
		});
	}

	$effect(() => {
		const wantRichInput = containsFileMentionLink(value ?? '') || containsCodeSpan(value ?? '');

		if (useRichInput === wantRichInput) return;

		if (!caretOffsetPinned) {
			pendingCaretOffset = inputRef?.getCaretOffset() ?? (value ?? '').length;
		}

		useRichInput = wantRichInput;
		queueCaretRestore();
	});

	$effect(() => {
		if (!value.trim()) dictated = false;
	});

	onMount(() => {
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
		speechRecognitionSupported = isSpeechRecognitionSupported();
		mediaRecorderSupported = typeof MediaRecorder !== 'undefined';
		dictationRecorder = new AudioRecorder();

		return () => {
			recognition?.abort();
			dictationRecorder?.cancelRecording();
		};
	});

	export function focus() {
		inputRef?.focus();
	}

	export function resetTextareaHeight() {
		inputRef?.resetHeight();
	}

	export function openModelSelector() {
		chatFormActionsRef?.openModelSelector();
	}

	/** True when speech filled the input; clears the mark and ends a running dictation. */
	export function takeDictated(): boolean {
		const wasDictated = dictated;

		dictated = false;
		dictationRun++;
		recognition?.abort();

		if (dictationRecorder?.isRecording()) {
			dictationRecorder.cancelRecording();
			isDictating = false;
		}

		return wasDictated;
	}

	export function checkModelSelected(): boolean {
		if (!hasModelSelected) {
			chatFormActionsRef?.openModelSelector();

			return false;
		}

		return true;
	}

	function handleFileSelect(files: File[]) {
		onFilesAdd?.(files);
	}

	function handleFileUpload() {
		fileInputRef?.click();
	}

	function handleFileRemove(fileId: string) {
		if (fileId.startsWith('attachment-')) {
			const index = parseInt(fileId.replace('attachment-', ''), 10);

			if (!isNaN(index) && index >= 0 && index < attachments.length) {
				onAttachmentRemove?.(index);
			}
		} else {
			onUploadedFileRemove?.(fileId);
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		// Pickers consume navigation/escape keys first; when consumed, skip
		// the enter-to-submit logic below.
		if (pickers.handleKeydown(event)) {
			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			const isModifier = event.ctrlKey || event.metaKey;
			const sendOnEnter = currentConfig.sendOnEnter !== false;

			// Caret inside a fenced code block (closed, or still open
			// while being typed): Enter adds a line, never submits. The
			// rich chat form input consumes this case locally; this gate
			// covers the plain textarea, where skipping submit lets the
			// native newline through.
			if (!isModifier && isOffsetInCodeBlock(value ?? '', inputRef?.getCaretOffset() ?? 0)) {
				return;
			}

			if (sendOnEnter || isModifier) {
				event.preventDefault();

				if (!canSubmit || disabled || hasLoadingAttachments) return;

				onSubmit?.();
			}
		}
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFilesAdd?.(files);

			return;
		}

		const text = event.clipboardData.getData(MimeTypeText.PLAIN);

		if (text.startsWith(CLIPBOARD_CONTENT_QUOTE_PREFIX)) {
			const parsed = parseClipboardContent(text);

			if (parsed.textAttachments.length > 0 || parsed.mcpPromptAttachments.length > 0) {
				event.preventDefault();
				value = parsed.message;
				onValueChange?.(parsed.message);

				// Handle text attachments as files
				if (parsed.textAttachments.length > 0) {
					const attachmentFiles = parsed.textAttachments.map(
						(att) =>
							new File([att.content], att.name, {
								type: MimeTypeText.PLAIN
							})
					);

					onFilesAdd?.(attachmentFiles);
				}

				// Handle MCP prompt attachments as ChatUploadedFile with mcpPrompt data
				if (parsed.mcpPromptAttachments.length > 0) {
					const mcpPromptFiles: ChatUploadedFile[] = parsed.mcpPromptAttachments.map((att) => ({
						file: new File([att.content], `${att.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						}),
						id: uuid(),
						isLoading: false,
						mcpPrompt: {
							arguments: att.arguments,
							promptName: att.promptName,
							serverName: att.serverName
						},
						name: att.name,
						size: att.content.length,
						textContent: att.content,
						type: SpecialFileType.MCP_PROMPT
					}));

					uploadedFiles = [...uploadedFiles, ...mcpPromptFiles];
					onUploadedFilesChange?.(uploadedFiles);
				}

				setTimeout(() => {
					inputRef?.focus();
				}, 10);

				return;
			}
		}

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: MimeTypeText.PLAIN
			});

			onFilesAdd?.([textFile]);
		}
	}

	function handlePromptLoadStart(
		placeholderId: string,
		promptInfo: MCPPromptInfo,
		args?: Record<string, string>
	) {
		pickers.closePromptPicker();

		const promptName = promptInfo.title || promptInfo.name;
		const placeholder: ChatUploadedFile = {
			file: new File([], 'loading'),
			id: placeholderId,
			isLoading: true,
			mcpPrompt: {
				arguments: args ? { ...args } : undefined,
				promptName: promptInfo.name,
				serverName: promptInfo.serverName
			},
			name: promptName,
			size: INITIAL_FILE_SIZE,
			type: SpecialFileType.MCP_PROMPT
		};

		uploadedFiles = [...uploadedFiles, placeholder];
		onUploadedFilesChange?.(uploadedFiles);
		inputRef?.focus();
	}

	function handlePromptLoadComplete(placeholderId: string, result: GetPromptResult) {
		const promptText = result.messages
			?.map((msg: PromptMessage) => {
				if (typeof msg.content === 'string') {
					return msg.content;
				}

				if (msg.content.type === ContentPartType.TEXT) {
					return msg.content.text;
				}

				return '';
			})
			.filter(Boolean)
			.join(PROMPT_CONTENT_SEPARATOR);

		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId
				? {
						...f,
						file: new File([promptText], `${f.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						}),
						isLoading: false,
						size: promptText.length,
						textContent: promptText
					}
				: f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	function handlePromptLoadError(placeholderId: string, error: string) {
		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId ? { ...f, isLoading: false, loadError: error } : f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	// Deferred so the closing popover's focus scope tears down first -
	// bits-ui yanks a synchronous focus() back into the still-mounted popover.
	function refocusInput() {
		queueMicrotask(() => inputRef?.focus());
	}

	// A flow outside the chat, e.g. the models page returning to it, asks for the
	// composer back. Focus on a frame of its own, once the route has settled.
	$effect(() => {
		if (!uiStore.composerFocusRequested) return;

		uiStore.composerFocusRequested = false;

		setTimeout(() => inputRef?.focus(), 0);
	});

	// Splice the mention link in place of the `@<query>` token. Uses the
	// live cursor, not a stale snapshot - the token may have been edited.
	function handleMentionSelect(entry: FileMentionEntry) {
		const cursor = inputRef?.getCaretOffset() ?? value.length;
		const token = findMentionToken(value, cursor);

		if (!token) return;

		const built = buildMentionInsertion(entry, value, token);

		if (!built) return;

		// Pin the post-insertion caret BEFORE the swap effect runs;
		// otherwise the effect clobbers it with the textarea's selection
		// at promotion time (browser-dependent: usually reset to 0).
		pendingCaretOffset = built.caretOffset;
		caretOffsetPinned = true;

		value = built.newValue;
		onValueChange?.(built.newValue);

		// Already in rich chat form input mode: no renderer flip, so the swap
		// effect's caret restore never runs.
		if (useRichInput) {
			queueCaretRestore();
		}
	}

	function dictationPrefix(): string {
		return value.trim() ? `${value.trimEnd()} ` : '';
	}

	function setDictatedText(prefix: string, text: string) {
		value = prefix + text;
		onValueChange?.(value);
		dictated = true;

		tick().then(() => {
			const el = inputRef?.getElement();

			if (el instanceof HTMLTextAreaElement) autoResizeTextarea(el);
		});
	}

	function handleDictateClick() {
		if (useServerDictation) {
			void handleServerDictation();

			return;
		}

		if (recognition) {
			recognition.stop();

			return;
		}

		const rec = createSpeechRecognition();

		if (!rec) return;

		// stop a reply being read so the mic does not hear it; the tap also lets iOS speak the next reply
		speechStore.stop();
		speechStore.unlock();

		const prefix = dictationPrefix();

		rec.onresult = (event) => {
			const text = speechResultsToText(event.results);

			if (text) setDictatedText(prefix, text);
		};
		rec.onerror = (event) => {
			if (SPEECH_RECOGNITION_SILENT_ERRORS.includes(event.error)) return;

			toast.error(
				SPEECH_RECOGNITION_BLOCKED_ERRORS.includes(event.error)
					? SPEECH_MIC_BLOCKED_MESSAGE
					: `Dictation failed: ${event.error}`
			);
		};
		rec.onend = () => {
			if (recognition !== rec) return;

			recognition = null;
			isDictating = false;
		};

		try {
			rec.start();
			recognition = rec;
			isDictating = true;
		} catch (error) {
			console.error('Failed to start dictation:', error);
		}
	}

	// an empty setting sends the selected model to this server in router mode, nothing otherwise
	function getTranscriptionModel(): string | undefined {
		const model = String(currentConfig.transcriptionModel ?? '').trim();

		if (model) return model;

		if (String(currentConfig.transcriptionUrl ?? '').trim() || !isRouter) return undefined;

		return modelsStore.selectedModelName ?? conversationModel ?? undefined;
	}

	// tap to record, tap again to send the audio to the transcription endpoint
	async function handleServerDictation() {
		if (!dictationRecorder || isTranscribing) return;

		if (dictationRecorder.isRecording()) {
			const run = dictationRun;

			isDictating = false;
			isTranscribing = true;

			try {
				const audio = await dictationRecorder.stopRecording();
				const wav = await convertToWav(audio, {
					mono: true,
					sampleRate: TRANSCRIPTION_SAMPLE_RATE
				});
				const text = await TranscriptionService.transcribe(createAudioFile(wav), {
					language: navigator.language.split('-')[0],
					model: getTranscriptionModel(),
					url: String(currentConfig.transcriptionUrl ?? '')
				});

				if (run === dictationRun && text) setDictatedText(dictationPrefix(), text);
			} catch (error) {
				if (run === dictationRun) {
					toast.error(
						`Transcription failed: ${error instanceof Error ? error.message : String(error)}`
					);
				}
			} finally {
				isTranscribing = false;
			}

			return;
		}

		if (!isAudioRecordingSupported()) {
			toast.error(SPEECH_MIC_BLOCKED_MESSAGE);

			return;
		}

		speechStore.stop();
		speechStore.unlock();

		try {
			await dictationRecorder.startRecording();
			isDictating = true;
		} catch (error) {
			toast.error(error instanceof Error ? error.message : SPEECH_MIC_BLOCKED_MESSAGE);
		}
	}

	async function handleMicClick() {
		if (!audioRecorder || !recordingSupported) {
			console.warn('Audio recording not supported');

			return;
		}

		if (isRecording) {
			isRecording = false;
			try {
				const audioBlob = await audioRecorder.stopRecording();
				const wavBlob = await convertToWav(audioBlob);
				const audioFile = createAudioFile(wavBlob);

				onFilesAdd?.([audioFile]);
			} catch (error) {
				console.error('Failed to stop recording:', error);
			}
		} else {
			try {
				await audioRecorder.startRecording();
				isRecording = true;
			} catch (error) {
				console.error('Failed to start recording:', error);
			}
		}
	}
</script>

<ChatFormInputFileInputInvisible bind:this={fileInputRef} onFileSelect={handleFileSelect} />

<form
	class="relative grid {className}"
	onsubmit={(event) => {
		event.preventDefault();

		if (!canSubmit || disabled || hasLoadingAttachments) return;

		onSubmit?.();
	}}
>
	<ChatFormPickers
		bind:this={pickersRef}
		commandQuery={pickers.commandQuery}
		commands={pickers.availableCommands}
		isCommandPickerOpen={pickers.isCommandPickerOpen}
		isMentionPickerOpen={pickers.isMentionPickerOpen}
		isPromptPickerOpen={pickers.isPromptPickerOpen}
		{mentionAnchor}
		mentionQuery={pickers.mentionQuery}
		onCommandPickerClose={pickers.handleCommandPickerClose}
		onCommandSelect={pickers.handleCommandSelect}
		onMentionOpened={() => inputRef?.focus()}
		onMentionPickerClose={pickers.handleMentionPickerClose}
		onMentionSelect={handleMentionSelect}
		onPromptLoadComplete={handlePromptLoadComplete}
		onPromptLoadError={handlePromptLoadError}
		onPromptLoadStart={handlePromptLoadStart}
		onPromptPickerClose={pickers.handlePromptPickerClose}
		promptSearchQuery={pickers.promptSearchQuery}
		scopePath={pickers.mentionScopePath}
	/>

	<div
		bind:this={mentionAnchor}
		aria-hidden="true"
		class="pointer-events-none absolute top-0 right-0 left-0 h-px"
	></div>

	<div
		class="{INPUT_CLASSES} overflow-hidden rounded-4xl md:rounded-3xl backdrop-blur-md {disabled
			? 'cursor-not-allowed opacity-60'
			: ''}"
		data-slot="input-area"
	>
		<ChatAttachmentsList
			bind:uploadedFiles
			activeModelId={activeModelId ?? undefined}
			{attachments}
			class="py-5"
			limitToSingleRow
			onFileRemove={handleFileRemove}
			style="scroll-padding: 1rem;"
		/>

		<div
			class="flex-column relative min-h-12 items-center rounded-4xl md:rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:py-3!"
		>
			<ChatFormInput
				bind:this={inputRef}
				bind:value
				class="px-5 py-1.5 md:pt-0"
				{disabled}
				onInput={() => {
					pickers.handleInput();
					onValueChange?.(value);
				}}
				onKeydown={handleKeydown}
				onPaste={handlePaste}
				{placeholder}
				{useRichInput}
			/>

			{#if mcpStore.resources.hasAttachments}
				<ChatFormMcpResourcesList
					class="mb-3"
					onResourceClick={(uri) => {
						preSelectedResourceUri = uri;
						isResourceDialogOpen = true;
					}}
				/>
			{/if}

			<ChatFormActions
				bind:this={chatFormActionsRef}
				canSend={canSubmit}
				class="px-3"
				{disabled}
				{isDictating}
				{isLoading}
				isReasoning={chatStore.isReasoning}
				{isRecording}
				{isTranscribing}
				onDictateClick={handleDictateClick}
				onFileUpload={handleFileUpload}
				onMicClick={handleMicClick}
				{onStop}
				onSystemPromptClick={() => onSystemPromptClick?.({ files: uploadedFiles, message: value })}
				{showAddButton}
				showDictateButton={useServerDictation ? mediaRecorderSupported : speechRecognitionSupported}
				{showModelSelector}
				{uploadedFiles}
			/>
		</div>
	</div>

	<ContextGaugePopup />

	{#if conversationsStore.preferences.hasEnabledCwdTools()}
		<ChatFormCurrentWorkingDirectory
			bind:query={pickers.workingDirectoryQuery}
			customAnchor={mentionAnchor}
			directory={cwd}
			{disabled}
			isOpen={pickers.isWorkingDirectoryPickerOpen}
			onChange={handleWorkingDirectoryChange}
			onClose={pickers.handleWorkingDirectoryClose}
			onOpen={pickers.handleWorkingDirectoryOpen}
		/>
	{/if}
</form>

<DialogMcpResourcesBrowser
	bind:open={isResourceDialogOpen}
	onAttach={(resource: MCPResourceInfo) => {
		mcpStore.attachResource(resource.uri);
	}}
	onOpenChange={(newOpen: boolean) => {
		if (!newOpen) {
			preSelectedResourceUri = undefined;
		}
	}}
	preSelectedUri={preSelectedResourceUri}
/>
