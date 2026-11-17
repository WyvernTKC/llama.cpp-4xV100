/**
 * AgenticStore - Multi-turn agentic loop orchestration
 *
 * Drives the agentic loop over MCP tools: streams each LLM turn, detects
 * tool calls, executes them via mcpStore, and enforces the turn limit. Each
 * turn produces one assistant message (with tool_calls) and one tool result
 * message per executed call, persisted as separate DB rows.
 *
 * Uses ChatService for streaming and mcpStore for tool execution; waits on
 * the permission/continue/steering gates owned by {@link AgenticGates}.
 */

import {
	AGENT_TOOL_NAMES,
	AGENT_TOOLS,
	CHAT_TOOL_NAMES,
	CHAT_TOOLS,
	DEFAULT_AGENTIC_CONFIG,
	NEWLINE,
	SUBAGENT_DISABLED_TOOLS
} from '#lib/constants/index.ts';
import {
	AUDIO_MIME_TO_EXTENSION,
	DATA_URI_BASE64_REGEX,
	DEFAULT_AUDIO_EXTENSION,
	DEFAULT_IMAGE_EXTENSION,
	DEFAULT_VIDEO_EXTENSION,
	IMAGE_MIME_TO_EXTENSION,
	MCP_ATTACHMENT_NAME_PREFIX,
	MIME_TYPE_PREFIXES,
	VIDEO_MIME_TO_EXTENSION
} from '#lib/constants/index.ts';
import { BuiltInTool, ToolPermissionDecision, ToolSource } from '#lib/enums/index.ts';
import {
	AttachmentType,
	ContentPartType,
	MessageRole,
	MessageType,
	MimeTypePrefix,
	ToolCallType
} from '#lib/enums/index.ts';
import { ChatHistoryService } from '#lib/services/chat-history.service.ts';
import { ChatService } from '#lib/services/index.ts';
import { MemoryService } from '#lib/services/memory.service.ts';
import { NotificationService } from '#lib/services/notification.service.ts';
import { ReadMediaService } from '#lib/services/read-media.service.ts';
import { SandboxService } from '#lib/services/sandbox.service.ts';
import { SearchChatsService } from '#lib/services/search-chats.service.ts';
import { ToolsService } from '#lib/services/tools.service.ts';
// direct imports between stores, not via the barrel, to avoid circular deps
import { AgenticGates } from '#lib/stores/agentic/gates.svelte.ts';
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';
import { mcpStore } from '#lib/stores/mcp/index.svelte.ts';
import { modelsStore } from '#lib/stores/models/index.svelte.ts';
import { settingsStore } from '#lib/stores/settings/index.svelte.ts';
import { toolsStore } from '#lib/stores/tools.svelte.ts';
import { wakeupsStore } from '#lib/stores/wakeups.svelte.ts';
import type {
	AgenticChatToolRequests,
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	AgenticMessage,
	AgenticToolCallList,
	AgenticToolPolicy,
	SteeringMessage
} from '#lib/types/agentic.js';
import type {
	ApiChatCompletionToolCall,
	ApiChatMessageContentPart,
	ApiChatMessageData
} from '#lib/types/api.js';
import type {
	ChatMessageAgenticTimings,
	ChatMessageAgenticTurnStats,
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatMessageToolCallTiming
} from '#lib/types/chat.js';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraVideoFile
} from '#lib/types/database.js';
import type {
	AgenticConfig,
	AgenticFlowParams,
	AgenticFlowResult,
	AgenticSession,
	MCPToolCall,
	MCPToolCallProgress,
	PendingQuestion,
	SettingsConfigType,
	ToolExecutionResult
} from '#lib/types/index.ts';
import type { OpenAIToolDefinition } from '#lib/types/mcp.js';
import {
	appendTextToContent,
	clampToolNumber,
	collectAttachmentRefs,
	contextReminderText,
	contextTokens,
	countSteps,
	describeTaskList,
	executeBrowserInfoTool,
	executeGetDatetimeTool,
	getAudioInputFormat,
	hasContextReminder,
	isAbortError,
	lastContextBearingMessage,
	normalizeAskArgs,
	normalizeConversationTitle,
	normalizeNotifyArgs,
	normalizeTaskList,
	parseMcpToolCallProgress,
	resolveAttachmentRefs,
	subagentPrompt,
	subagentTitle,
	truncateSubagentAnswer,
	uuid,
	withToolImageHandles
} from '#lib/utils/index.ts';
import { SvelteMap } from 'svelte/reactivity';

/** What the chat tools know about the conversation a flow started from. */
interface ChatToolFlowContext {
	/** Steps the model reads when the flow starts, what a compaction counts against. */
	baseSteps: number;
	/** The compaction epoch: the id of the last compaction node, empty before the first. */
	epoch: string;
	/** Context tokens of the last request before the flow, from the stored timings. */
	initialTokens: number;
	/** Whether the model was already reminded of the context size in this epoch. */
	reminded: boolean;
}

function isDatabaseMessage(
	message: AgenticFlowParams['messages'][number]
): message is DatabaseMessage {
	return 'convId' in message && typeof (message as DatabaseMessage).id === 'string';
}

function createDefaultSession(): AgenticSession {
	return {
		currentTurn: 0,
		executingToolCallId: null,
		flowRootMessageId: null,
		isRunning: false,
		lastError: null,
		liveLlm: null,
		pendingPermissionRequest: null,
		streamingToolCall: null,
		totalToolCalls: 0
	};
}

function toAgenticMessages(messages: ApiChatMessageData[]): AgenticMessage[] {
	return messages.map((message) => {
		if (
			message.role === MessageRole.ASSISTANT &&
			message.tool_calls &&
			message.tool_calls.length > 0
		) {
			return {
				content: message.content,
				reasoning_content: message.reasoning_content,
				role: MessageRole.ASSISTANT,
				tool_calls: message.tool_calls.map((call, index) => ({
					function: {
						arguments: call.function?.arguments ?? '',
						name: call.function?.name ?? ''
					},
					id: call.id ?? `call_${index}`,
					type: (call.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION
				}))
			} satisfies AgenticMessage;
		}

		if (message.role === MessageRole.ASSISTANT) {
			return {
				content: message.content,
				reasoning_content: message.reasoning_content,
				role: MessageRole.ASSISTANT
			} satisfies AgenticMessage;
		}

		if (message.role === MessageRole.TOOL && message.tool_call_id) {
			return {
				content: typeof message.content === 'string' ? message.content : '',
				role: MessageRole.TOOL,
				tool_call_id: message.tool_call_id
			} satisfies AgenticMessage;
		}

		return {
			content: message.content,
			role: message.role as MessageRole.SYSTEM | MessageRole.USER
		} satisfies AgenticMessage;
	});
}

class AgenticStore {
	// what the chat tools asked for during the running flow, per conversation
	private chatToolRequests = new Map<string, AgenticChatToolRequests>();
	// permission, continue and steering gates the loop waits on between turns
	private gates = new AgenticGates();
	// compaction epoch the context reminder went out in, per conversation
	private remindedEpochs = new Map<string, string>();
	private sessions = new SvelteMap<string, AgenticSession>();
	// the chat each run_subagent call runs in, by tool call id, for its block
	private subagentChats = new SvelteMap<string, string>();
	// what each running MCP tool call last reported, by tool call id, for its block
	private toolCallProgress = new SvelteMap<string, MCPToolCallProgress>();

	get isAnyRunning(): boolean {
		for (const session of this.sessions.values()) {
			if (session.isRunning) return true;
		}

		return false;
	}

	get isReady(): boolean {
		return true;
	}

	clearError(conversationId: string): void {
		this.updateSession(conversationId, { lastError: null });
	}

	clearSession(conversationId: string): void {
		this.sessions.delete(conversationId);
	}

	/**
	 * Clear the pending steering message without consuming it.
	 */
	clearSteeringMessage(conversationId: string): void {
		this.gates.clearSteeringMessage(conversationId);
	}

	constructor() {
		// drop per-conversation session state when the conversation is deleted,
		// otherwise every conversation that ever ran a flow leaks a session here
		conversationsStore.onConversationsDeleted((convIds) => {
			for (const convId of convIds) {
				this.sessions.delete(convId);
				this.chatToolRequests.delete(convId);
				this.remindedEpochs.delete(convId);
			}
		});
	}

	/**
	 * The chat tool requests of the flow that just ended, cleared on read.
	 * Called by chatStore after the agentic flow exits.
	 */
	consumeChatToolRequests(conversationId: string): AgenticChatToolRequests {
		const requests = this.chatToolRequests.get(conversationId) ?? {
			compact: false,
			renamed: false
		};

		this.chatToolRequests.delete(conversationId);

		return requests;
	}

	/**
	 * Consume and return the pending steering message for re-sending.
	 * Called by chatStore after the agentic flow exits.
	 */
	consumePendingSteeringMessage(conversationId: string): SteeringMessage | null {
		return this.gates.consumePendingSteeringMessage(conversationId);
	}

	getActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
		const active: Array<{ conversationId: string; session: AgenticSession }> = [];

		for (const [conversationId, session] of this.sessions.entries()) {
			if (session.isRunning) active.push({ conversationId, session });
		}

		return active;
	}

	getConfig(settings: SettingsConfigType): AgenticConfig {
		const maxTurns = Number(settings.agenticMaxTurns) || DEFAULT_AGENTIC_CONFIG.maxTurns;
		const hasTools =
			mcpStore.hasEnabledServers() ||
			toolsStore.serverTools.length > 0 ||
			toolsStore.browserTools.length > 0 ||
			toolsStore.customTools.length > 0;

		return {
			enabled: hasTools && DEFAULT_AGENTIC_CONFIG.enabled,
			maxTurns
		};
	}

	getCurrentTurn(conversationId: string): number {
		return this.sessions.get(conversationId)?.currentTurn ?? 0;
	}

	getExecutingToolCallId(conversationId: string): string | null {
		return this.sessions.get(conversationId)?.executingToolCallId ?? null;
	}

	// read-only: safe to call from derivations, unlike getSession
	getFlowRootMessageId(conversationId: string): string | null {
		return this.sessions.get(conversationId)?.flowRootMessageId ?? null;
	}

	getLastError(conversationId: string): Error | null {
		return this.sessions.get(conversationId)?.lastError ?? null;
	}

	// read-only: safe to call from derivations, unlike getSession
	getLiveLlmTotals(conversationId: string): AgenticSession['liveLlm'] {
		return this.sessions.get(conversationId)?.liveLlm ?? null;
	}

	getPendingContinueRequest(conversationId: string): boolean {
		return this.gates.getPendingContinueRequest(conversationId);
	}

	getPendingPermissionRequest(
		conversationId: string
	): { toolName: string; serverLabel: string } | null {
		return this.gates.getPendingPermissionRequest(conversationId);
	}

	/** The ask_user question the conversation waits on. */
	getPendingQuestion(conversationId: string): PendingQuestion | null {
		return this.gates.getPendingQuestion(conversationId);
	}

	getPendingSteeringMessageContent(conversationId: string): string | null {
		return this.gates.getPendingSteeringMessageContent(conversationId);
	}

	getPendingSteeringMessageExtras(conversationId: string): DatabaseMessageExtra[] | undefined {
		return this.gates.getPendingSteeringMessageExtras(conversationId);
	}

	getSession(conversationId: string): AgenticSession {
		let session = this.sessions.get(conversationId);

		if (!session) {
			session = createDefaultSession();
			this.sessions.set(conversationId, session);
		}

		return session;
	}

	getStreamingToolCall(conversationId: string): { name: string; arguments: string } | null {
		return this.sessions.get(conversationId)?.streamingToolCall ?? null;
	}

	/** The chat a run_subagent call runs in, while it runs and after. */
	getSubagentConversation(toolCallId: string): string | null {
		return this.subagentChats.get(toolCallId) ?? null;
	}

	/** The last progress a running MCP tool call reported, null when none. */
	getToolCallProgress(toolCallId: string): MCPToolCallProgress | null {
		return this.toolCallProgress.get(toolCallId) ?? null;
	}

	getTotalToolCalls(conversationId: string): number {
		return this.sessions.get(conversationId)?.totalToolCalls ?? 0;
	}

	/** Whether the running flow asked for a compaction once it ends. */
	hasCompactionRequest(conversationId: string): boolean {
		return this.chatToolRequests.get(conversationId)?.compact ?? false;
	}

	hasPendingSteeringMessage(conversationId: string): boolean {
		return this.gates.hasPendingSteeringMessage(conversationId);
	}

	/**
	 * Queue a steering message. When the current agentic turn completes,
	 * the flow exits and the caller re-sends the message as a normal chat message.
	 */
	injectSteeringMessage(
		conversationId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	): void {
		this.gates.injectSteeringMessage(conversationId, content, extras);
	}

	isRunning(conversationId: string): boolean {
		return this.sessions.get(conversationId)?.isRunning ?? false;
	}

	/**
	 * What a request of the chat sends under this tool policy: its messages
	 * as the agentic flow shapes them, and the tools it offers, none when the
	 * flow does not run. A request built from it shares the prefix the server
	 * cached for the conversation.
	 */
	async requestFor(
		messages: ApiChatMessageData[],
		toolPolicy?: AgenticToolPolicy
	): Promise<{ messages: ApiChatMessageData[]; tools: OpenAIToolDefinition[] }> {
		const tools = await this.toolsFor(toolPolicy);

		return {
			messages: tools.length > 0 ? (toAgenticMessages(messages) as ApiChatMessageData[]) : messages,
			tools
		};
	}

	resolveContinue(conversationId: string, shouldContinue: boolean): void {
		this.gates.resolveContinue(conversationId, shouldContinue);
	}

	resolvePermission(conversationId: string, decision: ToolPermissionDecision): void {
		this.gates.resolvePermission(conversationId, decision);
	}

	/** Answer the open ask_user question of a conversation; false when there is none. */
	resolveQuestion(conversationId: string, answer: string): boolean {
		return this.gates.resolveQuestion(conversationId, answer);
	}

	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		const {
			callbacks,
			conversationId,
			flowRootMessageId,
			messages,
			options = {},
			signal,
			toolPolicy
		} = params;

		// Clear any pending permissions/continue requests for this conversation when starting a new flow
		this.gates.clear(conversationId);
		this.chatToolRequests.delete(conversationId);

		const agenticConfig = this.getConfig(settingsStore.config);
		const tools = await this.toolsFor(toolPolicy);

		if (tools.length === 0) {
			return { handled: false };
		}

		const hasMcpServers = conversationsStore.preferences.policyEnabledServerIds().length > 0;

		console.log(`[AgenticStore] Starting agentic flow with ${tools.length} tools`);

		const normalizedMessages: ApiChatMessageData[] =
			await ChatService.normalizeMessagesForApi(messages);

		this.updateSession(conversationId, {
			currentTurn: 0,
			flowRootMessageId: flowRootMessageId ?? null,
			isRunning: true,
			lastError: null,
			liveLlm: null,
			totalToolCalls: 0
		});

		if (hasMcpServers) mcpStore.acquireConnection();

		try {
			await this.executeAgenticLoop({
				agenticConfig,
				attachmentRefs: collectAttachmentRefs(messages),
				callbacks,
				chatTools: this.chatToolFlowContext(conversationId, messages),
				conversationId,
				messages: normalizedMessages,
				options,
				signal,
				toolPolicy,
				tools
			});

			return { handled: true };
		} catch (error) {
			const normalizedError = error instanceof Error ? error : new Error(String(error));

			this.updateSession(conversationId, { lastError: normalizedError });
			callbacks.onError?.(normalizedError);

			return { error: normalizedError, handled: true };
		} finally {
			this.updateSession(conversationId, {
				flowRootMessageId: null,
				isRunning: false,
				liveLlm: null
			});

			if (hasMcpServers) {
				await mcpStore
					.releaseConnection()
					.catch((err: unknown) =>
						console.warn('[AgenticStore] Failed to release MCP connection:', err)
					);
			}
		}
	}

	private buildAttachmentName(mimeType: string, index: number): string {
		let extension = IMAGE_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_IMAGE_EXTENSION;

		if (mimeType.startsWith(MimeTypePrefix.AUDIO)) {
			extension = AUDIO_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_AUDIO_EXTENSION;
		} else if (mimeType.startsWith(MimeTypePrefix.VIDEO)) {
			extension = VIDEO_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_VIDEO_EXTENSION;
		}

		return `${MCP_ATTACHMENT_NAME_PREFIX}-${Date.now()}-${index}.${extension}`;
	}

	private buildFinalTimings(
		capturedTimings: ChatMessageTimings | undefined,
		agenticTimings: ChatMessageAgenticTimings
	): ChatMessageTimings | undefined {
		if (agenticTimings.toolCallsCount === 0) return capturedTimings;

		return {
			agentic: agenticTimings,
			cache_n: capturedTimings?.cache_n,
			predicted_ms: capturedTimings?.predicted_ms,
			predicted_n: capturedTimings?.predicted_n,
			prompt_ms: capturedTimings?.prompt_ms,
			prompt_n: capturedTimings?.prompt_n
		};
	}

	/** What the chat tools know about the conversation, from the messages a flow starts with. */
	private chatToolFlowContext(
		conversationId: string,
		messages: AgenticFlowParams['messages']
	): ChatToolFlowContext {
		const stored = messages.filter(isDatabaseMessage);

		let boundary: DatabaseMessage | undefined;

		for (let i = stored.length - 1; i >= 0 && !boundary; i--) {
			if (stored[i].type === MessageType.COMPACTION) boundary = stored[i];
		}

		const epoch = boundary?.id ?? '';
		const since = boundary?.timestamp ?? 0;
		// the steps a compaction keeps are older than its node, so only a reminder
		// stored after the node counts for this epoch
		const remindedInStore = stored.some(
			(m) =>
				m.timestamp > since && m.type !== MessageType.COMPACTION && hasContextReminder(m.content)
		);

		return {
			baseSteps: countSteps(stored),
			epoch,
			initialTokens: contextTokens(lastContextBearingMessage(stored)?.timings),
			reminded: remindedInStore || this.remindedEpochs.get(conversationId) === epoch
		};
	}

	/** The context size of the model a flow runs on, from its props or the server's. */
	private contextSize(model: string): number | null {
		return (model ? modelsStore.props.getModelContextSize(model) : null) ?? serverStore.contextSize;
	}

	private async executeAgenticLoop(params: {
		chatTools: ChatToolFlowContext;
		conversationId: string;
		messages: ApiChatMessageData[];
		options: AgenticFlowOptions;
		tools: ReturnType<typeof toolsStore.getEnabledToolsForLLM>;
		toolPolicy?: AgenticToolPolicy;
		agenticConfig: AgenticConfig;
		attachmentRefs: Map<string, string>;
		callbacks: AgenticFlowCallbacks;
		signal?: AbortSignal;
	}): Promise<void> {
		const {
			agenticConfig,
			attachmentRefs,
			callbacks,
			chatTools,
			conversationId,
			messages,
			options,
			signal,
			toolPolicy,
			tools
		} = params;
		const {
			createAssistantMessage,
			createToolResultMessage,
			onAssistantTurnComplete,
			onChunk,
			onCompletionId,
			onFlowComplete,
			onModel,
			onReasoningChunk,
			onTimings,
			onToolCallsStreaming,
			onTurnComplete,
			updateToolResultMessage
		} = callbacks;
		const sessionMessages: AgenticMessage[] = toAgenticMessages(messages);

		let capturedTimings: ChatMessageTimings | undefined;
		let totalToolCallCount = 0;

		const agenticTimings: ChatMessageAgenticTimings = {
			llm: { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 },
			perTurn: [],
			toolCalls: [],
			toolCallsCount: 0,
			toolsMs: 0,
			turns: 0
		};
		const maxTurns = agenticConfig.maxTurns;
		const effectiveModel = options.model || modelsStore.models[0]?.model || '';

		let turn = 0;
		// the last tool result of the previous turn, where a context reminder is stored
		let lastToolResult: { content: string; id: string } | null = null;

		while (true) {
			if (turn >= maxTurns) {
				// Turn limit reached - ask user whether to continue
				const shouldContinue = await this.gates.requestContinue(conversationId, signal);

				// Yield to allow Svelte to flush the UI update
				await new Promise((r) => setTimeout(r, 0));

				if (!shouldContinue || signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				// User chose to continue - extend the limit
				turn = 0;
			}

			this.updateSession(conversationId, { currentTurn: turn + 1 });
			agenticTimings.turns = turn + 1;

			if (signal?.aborted) {
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			// For turns > 0, create a new assistant message via callback
			if (turn > 0 && createAssistantMessage) {
				await createAssistantMessage();
			}

			let turnContent = '';
			let turnReasoningContent = '';
			let turnToolCalls: ApiChatCompletionToolCall[] = [];
			let lastStreamingToolCallName = '';
			let lastStreamingToolCallArgsLength = 0;
			let turnTimings: ChatMessageTimings | undefined;

			const turnStats: ChatMessageAgenticTurnStats = {
				llm: { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 },
				toolCalls: [],
				toolsMs: 0,
				turn: turn + 1
			};

			await this.remindContextIfFull({
				chatTools,
				conversationId,
				lastToolResult,
				model: effectiveModel,
				sessionMessages,
				tokens: capturedTimings ? contextTokens(capturedTimings) : chatTools.initialTokens,
				tools,
				updateToolResultMessage
			});
			lastToolResult = null;

			try {
				await ChatService.sendMessage(
					sessionMessages as ApiChatMessageData[],
					{
						...options,
						onChunk: (chunk: string) => {
							turnContent += chunk;
							onChunk?.(chunk);
						},
						onComplete: () => {
							/* Completion handled after sendMessage resolves */
						},
						onCompletionId,
						onError: (error: Error) => {
							throw error;
						},
						onModel,
						onReasoningChunk: (chunk: string) => {
							turnReasoningContent += chunk;
							onReasoningChunk?.(chunk);
						},
						onTimings: (timings?: ChatMessageTimings, progress?: ChatMessagePromptProgress) => {
							onTimings?.(timings, progress);

							if (timings) {
								capturedTimings = timings;
								turnTimings = timings;

								// completed turns + in-flight turn live counts
								this.updateSession(conversationId, {
									liveLlm: {
										predicted_ms: agenticTimings.llm.predicted_ms + (timings.predicted_ms ?? 0),
										predicted_n: agenticTimings.llm.predicted_n + (timings.predicted_n ?? 0),
										prompt_ms: agenticTimings.llm.prompt_ms + (timings.prompt_ms ?? 0),
										prompt_n: agenticTimings.llm.prompt_n + (timings.prompt_n ?? 0)
									}
								});
							}
						},
						onToolCallChunk: (serialized: string) => {
							try {
								turnToolCalls = JSON.parse(serialized) as ApiChatCompletionToolCall[];

								onToolCallsStreaming?.(turnToolCalls);

								if (turnToolCalls.length > 0 && turnToolCalls[0]?.function) {
									const name = turnToolCalls[0].function.name || '';
									const args = turnToolCalls[0].function.arguments || '';
									const argsLengthBucket = Math.floor(args.length / 100);

									if (
										name !== lastStreamingToolCallName ||
										argsLengthBucket !== lastStreamingToolCallArgsLength
									) {
										lastStreamingToolCallName = name;
										lastStreamingToolCallArgsLength = argsLengthBucket;
										this.updateSession(conversationId, {
											streamingToolCall: { arguments: args, name }
										});
									}
								}
							} catch {
								/* Ignore parse errors during streaming */
							}
						},
						stream: true,
						tools: tools.length > 0 ? tools : undefined
					},
					conversationId,
					signal
				);

				this.updateSession(conversationId, { streamingToolCall: null });

				if (turnTimings) {
					agenticTimings.llm.predicted_n += turnTimings.predicted_n || 0;
					agenticTimings.llm.predicted_ms += turnTimings.predicted_ms || 0;
					agenticTimings.llm.prompt_n += turnTimings.prompt_n || 0;
					agenticTimings.llm.prompt_ms += turnTimings.prompt_ms || 0;
					turnStats.llm.predicted_n = turnTimings.predicted_n || 0;
					turnStats.llm.predicted_ms = turnTimings.predicted_ms || 0;
					turnStats.llm.prompt_n = turnTimings.prompt_n || 0;
					turnStats.llm.prompt_ms = turnTimings.prompt_ms || 0;
				}
			} catch (error) {
				if (signal?.aborted) {
					// Save whatever we have for this turn before exiting
					await onAssistantTurnComplete?.(
						turnContent,
						turnReasoningContent || undefined,
						this.buildFinalTimings(capturedTimings, agenticTimings),
						undefined
					);
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				const normalizedError = error instanceof Error ? error : new Error('LLM stream error');

				// preserve partial output as is, the outer error dialog informs the user separately
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				throw normalizedError;
			}

			// If the abort landed while ChatService.sendMessage was still resolving, the
			// outer catch above never fires because ChatService swallows the AbortError
			// and returns normally. Bail out here so a half-received tool_call (truncated
			// arguments JSON) is not persisted as if it were complete.
			if (signal?.aborted) {
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			// === Steering check: if a user message was queued during this turn, exit the flow.
			// The caller (chatStore) will consume the pending message and re-send it normally.
			if (this.gates.hasPendingSteeringMessage(conversationId)) {
				console.log('[AgenticStore] Steering message detected after turn, exiting agentic flow');
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					turnToolCalls.length > 0 ? this.normalizeToolCalls(turnToolCalls) : undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			// No tool calls = final turn, save and complete
			if (turnToolCalls.length === 0) {
				agenticTimings.perTurn!.push(turnStats);

				const finalTimings = this.buildFinalTimings(capturedTimings, agenticTimings);

				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					finalTimings,
					undefined
				);

				if (finalTimings) onTurnComplete?.(finalTimings);

				onFlowComplete?.(finalTimings);

				return;
			}

			// Normalize and save assistant turn with tool calls. Unless parallel tool
			// calls are on, the request asks for one call per reply; a template that
			// still yields several keeps only the first, so the model reads its result
			// before it decides on the next call
			let normalizedCalls = this.normalizeToolCalls(turnToolCalls);

			if (!options.parallelToolCalls && normalizedCalls.length > 1) {
				console.info(
					`[AgenticStore] Running 1 of ${normalizedCalls.length} tool calls, parallel tool calls are off`
				);
				normalizedCalls = normalizedCalls.slice(0, 1);
			}

			if (normalizedCalls.length === 0) {
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			totalToolCallCount += normalizedCalls.length;
			this.updateSession(conversationId, { totalToolCalls: totalToolCallCount });

			// Save the assistant message with its tool calls
			await onAssistantTurnComplete?.(
				turnContent,
				turnReasoningContent || undefined,
				turnTimings,
				normalizedCalls
			);

			// Add assistant message to session history
			sessionMessages.push({
				content: turnContent || undefined,
				reasoning_content: turnReasoningContent || undefined,
				role: MessageRole.ASSISTANT,
				tool_calls: normalizedCalls
			});

			// Execute each tool call and create result messages
			for (let i = 0; i < normalizedCalls.length; i++) {
				const toolCall = normalizedCalls[i];

				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				// Check for pending steering message - skip remaining tool calls
				if (this.gates.hasPendingSteeringMessage(conversationId)) {
					console.log(
						`[AgenticStore] Steering message detected, skipping ${normalizedCalls.length - i} remaining tool call(s)`
					);
					for (let j = i; j < normalizedCalls.length; j++) {
						const remainingCall = normalizedCalls[j];
						const interruptedContent = 'Tool execution was interrupted by a new user message.';

						if (createToolResultMessage) {
							await createToolResultMessage(remainingCall.id, interruptedContent);
						}

						sessionMessages.push({
							content: interruptedContent,
							role: MessageRole.TOOL,
							tool_call_id: remainingCall.id
						});
					}

					break;
				}

				const toolName = toolCall.function.name;
				const serverLabel = toolsStore.getToolServerLabel(toolName);
				// Ask for permission before executing the tool
				const permission = await this.gates.requestPermission(
					conversationId,
					toolName,
					serverLabel,
					signal
				);

				// Yield to allow Svelte to flush the UI update (hide permission dialog)
				await new Promise((r) => setTimeout(r, 0));

				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				const toolStartTime = performance.now();
				const toolSource = toolsStore.getToolSource(toolName);

				let result = '';
				let toolSuccess = true;
				let createdToolResultMessageId: string | null = null;
				let linkedVideos: DatabaseMessageExtraVideoFile[] = [];

				// Streaming tools (currently only exec_shell_command): mark
				// the session so the matching renderer can switch to live mode.
				// Cleared unconditionally below.
				this.updateSession(conversationId, { executingToolCallId: toolCall.id });

				if (permission === ToolPermissionDecision.DENY) {
					result = 'Tool execution was denied by the user.';
					toolSuccess = false;
				} else {
					try {
						if (
							toolSource === ToolSource.SERVER &&
							toolName === BuiltInTool.SERVER_EXEC_SHELL_COMMAND &&
							createToolResultMessage &&
							updateToolResultMessage
						) {
							const args = this.parseToolArguments(toolCall.function.arguments);
							const cwd = conversationsStore.activeConversation?.cwd;
							const msg = await createToolResultMessage(toolCall.id, '', undefined, cwd);

							createdToolResultMessageId = msg.id;

							let accumulated = '';

							for await (const ev of ToolsService.streamTool(toolName, args, signal, cwd)) {
								if (ev.chunk !== null) {
									accumulated += ev.chunk;
									await updateToolResultMessage(msg.id, accumulated);
								}

								if (ev.done) {
									if (ev.error) {
										accumulated = accumulated
											? `${accumulated}\nError: ${ev.error}`
											: `Error: ${ev.error}`;
										await updateToolResultMessage(msg.id, accumulated);
										toolSuccess = false;
									}

									break;
								}
							}
							result = accumulated;
						} else if (toolSource === ToolSource.SERVER) {
							const args = this.parseToolArguments(toolCall.function.arguments);
							const cwd = conversationsStore.activeConversation?.cwd;
							const executionResult = await ToolsService.executeTool(toolName, args, signal, cwd);

							result = executionResult.content;

							if (executionResult.isError) toolSuccess = false;
						} else if (toolSource === ToolSource.BROWSER) {
							const args = this.parseToolArguments(toolCall.function.arguments);

							let executionResult: ToolExecutionResult;

							if (toolName === BuiltInTool.BROWSER_GET_DATETIME) {
								executionResult = executeGetDatetimeTool();
							} else if (toolName === BuiltInTool.SERVER_GET_INFO) {
								executionResult = executeBrowserInfoTool();
							} else if (toolName === BuiltInTool.BROWSER_SEARCH_CHATS) {
								executionResult = await SearchChatsService.executeTool(args, conversationId);
							} else if (toolName === BuiltInTool.BROWSER_READ_MEDIA) {
								executionResult = await ReadMediaService.executeTool(
									args,
									{
										audio: modelsStore.props.modelSupportsAudio(effectiveModel),
										vision: modelsStore.props.modelSupportsVision(effectiveModel)
									},
									signal,
									conversationsStore.activeConversation?.cwd
								);
							} else if (CHAT_TOOL_NAMES.includes(toolName)) {
								executionResult = await this.executeChatTool(toolName, args, {
									chatTools,
									conversationId,
									model: effectiveModel,
									sendInBackground: callbacks.sendInBackground,
									timings: capturedTimings,
									turn
								});
							} else if (AGENT_TOOL_NAMES.includes(toolName)) {
								executionResult = await this.executeAgentTool(toolName, args, {
									conversationId,
									sendInBackground: callbacks.sendInBackground,
									signal,
									toolCallId: toolCall.id,
									toolPolicy
								});
							} else {
								executionResult = await SandboxService.executeTool(toolName, args, signal);
							}

							result = executionResult.content;

							if (executionResult.isError) toolSuccess = false;
						} else {
							const mcpCall: MCPToolCall = {
								function: {
									arguments: resolveAttachmentRefs(
										toolCall.function.arguments,
										tools.find((tool) => tool.function.name === toolName)?.function.parameters,
										attachmentRefs
									),
									name: toolName
								},
								id: toolCall.id
							};

							try {
								const executionResult = await mcpStore.executeTool(mcpCall, signal, (progress) => {
									this.toolCallProgress.set(
										toolCall.id,
										parseMcpToolCallProgress(progress, this.toolCallProgress.get(toolCall.id))
									);
								});

								result = executionResult.content;
							} finally {
								this.toolCallProgress.delete(toolCall.id);
							}

							// shown in the chat only: the model keeps reading the links
							linkedVideos = await mcpStore.fetchToolResultVideos(toolName, result, signal);
						}
					} catch (error) {
						if (isAbortError(error)) {
							this.updateSession(conversationId, { executingToolCallId: null });
							onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

							return;
						}

						// Carry the partial stream contents already mirrored to the UI -
						// they show up as live output even if the stream broke off mid-run.
						result = result
							? `${result}\nError: ${error instanceof Error ? error.message : String(error)}`
							: `Error: ${error instanceof Error ? error.message : String(error)}`;
						toolSuccess = false;

						if (createdToolResultMessageId && updateToolResultMessage) {
							await updateToolResultMessage(createdToolResultMessageId, result);
						}
					}
				}

				this.updateSession(conversationId, { executingToolCallId: null });

				const toolDurationMs = performance.now() - toolStartTime;
				const toolTiming: ChatMessageToolCallTiming = {
					duration_ms: Math.round(toolDurationMs),
					name: toolCall.function.name,
					success: toolSuccess
				};

				agenticTimings.toolCalls!.push(toolTiming);
				agenticTimings.toolCallsCount++;
				agenticTimings.toolsMs += Math.round(toolDurationMs);
				turnStats.toolCalls.push(toolTiming);
				turnStats.toolsMs += Math.round(toolDurationMs);

				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				const { attachments, cleanedResult } = this.extractBase64Attachments(result);

				attachments.push(...linkedVideos);

				// a later call in this flow can pass these images on by their handles
				for (const attachment of attachments) {
					if (attachment.type === AttachmentType.IMAGE && attachment.refId) {
						attachmentRefs.set(attachment.refId, attachment.base64Url);
					}
				}

				// the text the model reads, as it reads it again from the stored message
				const modelResult = withToolImageHandles(cleanedResult, attachments);

				// For streaming tools the result message was created empty
				// at the start of execution and updated in place as chunks
				// arrived via updateToolResultMessage. Skip the second
				// create call - just attach any base64 attachments found in
				// the final accumulator (rare, since chunks usually don't
				// carry image data URIs). Both paths store the extras once.
				if (createdToolResultMessageId) {
					if (attachments.length > 0 && updateToolResultMessage) {
						await updateToolResultMessage(createdToolResultMessageId, cleanedResult, attachments);
					}

					lastToolResult = { content: cleanedResult, id: createdToolResultMessageId };
				} else if (createToolResultMessage) {
					const created = await createToolResultMessage(
						toolCall.id,
						cleanedResult,
						attachments.length > 0 ? attachments : undefined
					);

					lastToolResult = { content: cleanedResult, id: created.id };
				}

				// Build content parts for session history (including images for vision models)
				const contentParts: ApiChatMessageContentPart[] = [
					{ text: modelResult, type: ContentPartType.TEXT }
				];

				for (const attachment of attachments) {
					if (attachment.type === AttachmentType.AUDIO) {
						if (modelsStore.props.modelSupportsAudio(effectiveModel)) {
							contentParts.push({
								input_audio: {
									data: (attachment as DatabaseMessageExtraAudioFile).base64Data,
									format: getAudioInputFormat(
										(attachment as DatabaseMessageExtraAudioFile).mimeType
									)
								},
								type: ContentPartType.INPUT_AUDIO
							});
						}
					} else if (attachment.type === AttachmentType.IMAGE) {
						if (modelsStore.props.modelSupportsVision(effectiveModel)) {
							contentParts.push({
								image_url: {
									url: (attachment as DatabaseMessageExtraImageFile).base64Url
								},
								type: ContentPartType.IMAGE_URL
							});
						} else {
							console.info(
								`[AgenticStore] Skipping image attachment (model "${effectiveModel}" does not support vision)`
							);
						}
					}
				}

				sessionMessages.push({
					content: contentParts.length === 1 ? modelResult : contentParts,
					role: MessageRole.TOOL,
					tool_call_id: toolCall.id
				});
			}

			if (turnStats.toolCalls.length > 0) {
				agenticTimings.perTurn!.push(turnStats);

				const intermediateTimings = this.buildFinalTimings(capturedTimings, agenticTimings);

				if (intermediateTimings) onTurnComplete?.(intermediateTimings);
			}

			// If tools were interrupted by a steering message, exit now instead of starting another LLM turn
			if (this.gates.hasPendingSteeringMessage(conversationId)) {
				console.log(
					'[AgenticStore] Steering message detected after tool execution, exiting agentic flow'
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			turn++;
		}
	}

	/**
	 * Runs an agent tool: sleep, ask_user, update_tasks, notify,
	 * schedule_wakeup and run_subagent. A stop through `signal` while one
	 * waits throws an abort, which ends the flow like any stopped tool.
	 */
	private async executeAgentTool(
		toolName: string,
		args: Record<string, unknown>,
		ctx: {
			conversationId: string;
			sendInBackground?: AgenticFlowCallbacks['sendInBackground'];
			signal?: AbortSignal;
			toolCallId: string;
			toolPolicy?: AgenticToolPolicy;
		}
	): Promise<ToolExecutionResult> {
		const { conversationId, signal } = ctx;

		switch (toolName) {
			case BuiltInTool.BROWSER_SLEEP: {
				const seconds = clampToolNumber(args.seconds, AGENT_TOOLS.SLEEP_MAX_SECONDS);
				const { slept, woken } = await this.sleep(conversationId, seconds, signal);

				return {
					content: woken
						? `Woke up after ${slept} of ${seconds} seconds: the user sent a message.`
						: `Waited ${slept} seconds.`,
					isError: false
				};
			}

			case BuiltInTool.BROWSER_ASK_USER: {
				const ask = normalizeAskArgs(args);

				if (!ask) {
					return {
						content: 'Error: ask_user needs a question and at least two different options.',
						isError: true
					};
				}

				const answer = await this.gates.requestAnswer(
					conversationId,
					{ ...ask, toolCallId: ctx.toolCallId },
					signal
				);

				if (signal?.aborted) throw new DOMException('Stopped', 'AbortError');

				if (answer === null) {
					return { content: 'The question was closed without an answer.', isError: false };
				}

				return { content: JSON.stringify({ answer }), isError: false };
			}

			case BuiltInTool.BROWSER_UPDATE_TASKS: {
				const tasks = normalizeTaskList(args);

				if (!tasks) {
					return {
						content: 'Error: update_tasks needs tasks, a list of { content, status }.',
						isError: true
					};
				}

				return { content: describeTaskList(tasks), isError: false };
			}

			case BuiltInTool.BROWSER_NOTIFY: {
				const notice = normalizeNotifyArgs(args);

				if (!notice) return { content: 'Error: the message is empty.', isError: true };

				const shown = await NotificationService.show(notice.title, notice.message);

				return {
					content:
						shown === 'system'
							? 'Notification shown.'
							: 'Shown inside the page: this browser does not allow system notifications here.',
					isError: false
				};
			}

			case BuiltInTool.BROWSER_SCHEDULE_WAKEUP: {
				const minutes = clampToolNumber(args.minutes, AGENT_TOOLS.WAKEUP_MAX_MINUTES);
				const note = typeof args.note === 'string' ? args.note.trim() : '';

				if (minutes === 0) {
					return {
						content: wakeupsStore.cancel(conversationId)
							? 'Wake-up cancelled.'
							: 'There was no wake-up to cancel.',
						isError: false
					};
				}

				if (!note) return { content: 'Error: the note is empty.', isError: true };

				const wakeup = wakeupsStore.schedule(conversationId, minutes, note);
				const at = new Date(wakeup.dueAt).toLocaleTimeString([], {
					hour: '2-digit',
					minute: '2-digit'
				});

				return {
					content: `Wake-up scheduled for ${at}, in ${Math.round(minutes * 10) / 10} minutes. It replaces any earlier one and fires only while the page is open.`,
					isError: false
				};
			}

			case BuiltInTool.BROWSER_RUN_SUBAGENT:
				return this.runSubagent(args, ctx);

			default:
				return { content: `Error: unknown agent tool ${toolName}.`, isError: true };
		}
	}

	/**
	 * Runs a chat tool: compaction, context usage, rename, search and read past
	 * conversations, new conversation, and the memory tools. A compaction is
	 * only requested here; chatStore runs it once the flow ends, since the loop
	 * holds the message list the model reads until then.
	 */
	private async executeChatTool(
		toolName: string,
		args: Record<string, unknown>,
		ctx: {
			chatTools: ChatToolFlowContext;
			conversationId: string;
			model: string;
			sendInBackground?: AgenticFlowCallbacks['sendInBackground'];
			timings?: ChatMessageTimings;
			turn: number;
		}
	): Promise<ToolExecutionResult> {
		const { chatTools, conversationId } = ctx;
		const requests = this.chatToolRequests.get(conversationId) ?? {
			compact: false,
			renamed: false
		};
		const preserve = Math.max(0, Number(settingsStore.config.compactionPreservedSteps) || 0);
		// steps once the reply ends: what the model read, this turn and the final answer
		const stepsAtEnd = chatTools.baseSteps + ctx.turn + 2;
		const canCompact = stepsAtEnd > preserve;

		switch (toolName) {
			case BuiltInTool.BROWSER_GET_CONTEXT_USAGE: {
				const used = ctx.timings ? contextTokens(ctx.timings) : chatTools.initialTokens;
				const total = this.contextSize(ctx.model);

				return {
					content: JSON.stringify({
						compaction_available: canCompact,
						compaction_keeps_last_steps: preserve,
						compaction_scheduled: requests.compact,
						context_size: total,
						percent_used: total ? Math.round((used / total) * 100) : null,
						used_tokens: used
					}),
					isError: false
				};
			}

			case BuiltInTool.BROWSER_COMPACT_CONVERSATION: {
				if (!canCompact) {
					return {
						content: `Nothing to compact yet: compaction keeps the last ${preserve} steps as written and this conversation has ${stepsAtEnd}.`,
						isError: false
					};
				}

				this.chatToolRequests.set(conversationId, { ...requests, compact: true });

				return {
					content: `Compaction scheduled. When this reply ends, everything before the last ${preserve} steps is replaced by a summary in your context, and the user still sees the full history. Finish your reply normally.`,
					isError: false
				};
			}

			case BuiltInTool.BROWSER_RENAME_CONVERSATION: {
				const title = normalizeConversationTitle(args.title);

				if (!title) return { content: 'Error: the title is empty.', isError: true };

				await conversationsStore.updateConversationName(conversationId, title);
				this.chatToolRequests.set(conversationId, { ...requests, renamed: true });

				return { content: `Renamed the conversation to "${title}".`, isError: false };
			}

			case BuiltInTool.BROWSER_SEARCH_CONVERSATIONS: {
				const { hits, partial } = await ChatHistoryService.search(args.query, conversationId);
				const note =
					hits.length === 0
						? 'No other conversation matches.'
						: partial
							? 'No conversation holds every word; these hold some of them.'
							: undefined;

				return { content: JSON.stringify({ conversations: hits, note }), isError: false };
			}

			case BuiltInTool.BROWSER_READ_CONVERSATION: {
				const id = typeof args.conversation_id === 'string' ? args.conversation_id : '';
				const read = await ChatHistoryService.read(id, Number(args.offset) || 0);

				if (!read) return { content: `Error: no conversation with id "${id}".`, isError: true };

				return { content: JSON.stringify(read), isError: false };
			}

			case BuiltInTool.BROWSER_NEW_CONVERSATION: {
				const title = normalizeConversationTitle(args.title) || 'New chat';
				const firstMessage =
					typeof args.first_message === 'string' ? args.first_message.trim() : '';
				const created = await conversationsStore.createConversationInBackground(title);
				const where = settingsStore.config.conversationTabs ? 'in a new tab' : 'in the chat list';

				// the new chat answers on its own while this flow goes on
				if (firstMessage && ctx.sendInBackground) {
					ctx
						.sendInBackground(created.id, firstMessage)
						.catch((error: unknown) =>
							console.error('[AgenticStore] The new conversation failed to start:', error)
						);
				}

				return {
					content: JSON.stringify({
						conversation_id: created.id,
						note: firstMessage
							? `The chat is ${where} and answers its first message on its own.`
							: `The chat is ${where}.`,
						title
					}),
					isError: false
				};
			}

			case BuiltInTool.BROWSER_SAVE_MEMORY: {
				const { created, memory } = await MemoryService.save(args.content, conversationId);

				return {
					content: created ? `Saved memory ${memory.id}.` : `Already saved as memory ${memory.id}.`,
					isError: false
				};
			}

			case BuiltInTool.BROWSER_SEARCH_MEMORIES: {
				const found = await MemoryService.search(
					args.query,
					CHAT_TOOLS.SEARCH_MEMORIES_MAX_RESULTS
				);

				return {
					content: JSON.stringify({
						memories: found.map((m) => ({
							content: m.content,
							id: m.id,
							saved: new Date(m.updatedAt).toISOString().slice(0, 10)
						}))
					}),
					isError: false
				};
			}

			case BuiltInTool.BROWSER_DELETE_MEMORY: {
				const id = typeof args.id === 'string' ? args.id : '';
				const removed = await MemoryService.remove(id);

				if (!removed) return { content: `Error: no single memory with id "${id}".`, isError: true };

				return { content: `Deleted memory ${removed.id}: ${removed.content}`, isError: false };
			}

			default:
				return { content: `Error: unknown chat tool ${toolName}.`, isError: true };
		}
	}

	private extractBase64Attachments(result: string): {
		cleanedResult: string;
		attachments: DatabaseMessageExtra[];
	} {
		if (!result.trim()) {
			return { attachments: [], cleanedResult: result };
		}

		const lines = result.split(NEWLINE);
		const attachments: DatabaseMessageExtra[] = [];

		let attachmentIndex = 0;

		const cleanedLines = lines.map((line) => {
			const trimmedLine = line.trim();
			const match = trimmedLine.match(DATA_URI_BASE64_REGEX);

			if (!match) {
				return line;
			}

			const mimeType = match[1].toLowerCase();
			const base64Data = match[2];

			if (!base64Data) {
				return line;
			}

			attachmentIndex += 1;
			const name = this.buildAttachmentName(mimeType, attachmentIndex);

			if (mimeType.startsWith(MIME_TYPE_PREFIXES.IMAGE)) {
				attachments.push({
					base64Url: trimmedLine,
					name,
					refId: uuid(),
					type: AttachmentType.IMAGE
				});

				return `[Attachment saved: ${name}]`;
			}

			if (mimeType.startsWith(MimeTypePrefix.AUDIO)) {
				// audio extras hold the bare base64, the input_audio part has no room for a data URI
				attachments.push({
					base64Data,
					mimeType,
					name,
					type: AttachmentType.AUDIO
				});

				return `[Attachment saved: ${name}]`;
			}

			if (mimeType.startsWith(MimeTypePrefix.VIDEO)) {
				attachments.push({
					base64Data,
					mimeType,
					name,
					type: AttachmentType.VIDEO
				});

				return `[Attachment saved: ${name}]`;
			}

			return line;
		});

		return { attachments, cleanedResult: cleanedLines.join(NEWLINE) };
	}

	private normalizeToolCalls(toolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
		if (!toolCalls) return [];

		return toolCalls.map((call, index) => ({
			function: {
				arguments: call?.function?.arguments ?? '',
				name: call?.function?.name ?? ''
			},
			id: call?.id ?? `tool_${index}`,
			type: (call?.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION
		}));
	}

	private parseToolArguments(args: string | Record<string, unknown>): Record<string, unknown> {
		if (typeof args === 'object') return args;

		const trimmed = args.trim();

		if (trimmed === '') return {};

		return JSON.parse(trimmed) as Record<string, unknown>;
	}

	/**
	 * Tells the model once per compaction epoch that the context is filling up:
	 * one line appended to the last message of the request. On a tool result the
	 * line is stored too, so the next request reads the same prefix and the
	 * server keeps its cache; a user message keeps its text as the user wrote it.
	 */
	private async remindContextIfFull(p: {
		chatTools: ChatToolFlowContext;
		conversationId: string;
		lastToolResult: { content: string; id: string } | null;
		model: string;
		sessionMessages: AgenticMessage[];
		tokens: number;
		tools: OpenAIToolDefinition[];
		updateToolResultMessage?: AgenticFlowCallbacks['updateToolResultMessage'];
	}): Promise<void> {
		const percent = Number(settingsStore.config.contextReminderPercent);

		if (p.chatTools.reminded || !settingsStore.config.chatToolsEnabled || !(percent > 0)) return;

		if (!p.tools.some((t) => t.function.name === BuiltInTool.BROWSER_COMPACT_CONVERSATION)) return;

		const total = this.contextSize(p.model);

		if (!total || p.tokens * 100 < percent * total) return;

		const last = p.sessionMessages[p.sessionMessages.length - 1];

		if (!last || (last.role !== MessageRole.USER && last.role !== MessageRole.TOOL)) return;

		const text = contextReminderText(p.tokens, total);

		last.content = appendTextToContent(last.content, text);
		p.chatTools.reminded = true;
		this.remindedEpochs.set(p.conversationId, p.chatTools.epoch);

		if (last.role === MessageRole.TOOL && p.lastToolResult && p.updateToolResultMessage) {
			await p.updateToolResultMessage(
				p.lastToolResult.id,
				appendTextToContent(p.lastToolResult.content, text) as string
			);
		}
	}

	/**
	 * Hands a task to a sub-agent: a chat of its own, under this flow's tool
	 * policy minus the tools a sub-agent does not get, that answers the task
	 * while this flow waits. Only its final answer comes back.
	 */
	private async runSubagent(
		args: Record<string, unknown>,
		ctx: {
			conversationId: string;
			sendInBackground?: AgenticFlowCallbacks['sendInBackground'];
			signal?: AbortSignal;
			toolCallId: string;
			toolPolicy?: AgenticToolPolicy;
		}
	): Promise<ToolExecutionResult> {
		const task = typeof args.task === 'string' ? args.task.trim() : '';

		if (!task) return { content: 'Error: the task is empty.', isError: true };

		if (!ctx.sendInBackground) {
			return { content: 'Error: sub-agents cannot run from here.', isError: true };
		}

		const title = subagentTitle(args.title, task);
		const disabledTools = [...(ctx.toolPolicy?.disabledTools ?? toolsStore.disabledTools)];

		for (const name of SUBAGENT_DISABLED_TOOLS) {
			const key = toolsStore.getPermissionKey(name) ?? `${ToolSource.BROWSER}:${name}`;

			if (!disabledTools.includes(key)) disabledTools.push(key);
		}

		const chat = await conversationsStore.createConversationInBackground(title, {
			policy: {
				disabledToolCategories: [
					...(ctx.toolPolicy?.disabledToolCategories ?? toolsStore.disabledToolCategories)
				],
				disabledTools
			}
		});

		this.subagentChats.set(ctx.toolCallId, chat.id);

		const reply = await ctx.sendInBackground(chat.id, subagentPrompt(task), ctx.signal);

		if (ctx.signal?.aborted) throw new DOMException('Stopped', 'AbortError');

		if (!reply) {
			return {
				content: JSON.stringify({
					conversation_id: chat.id,
					error: 'The sub-agent did not start.'
				}),
				isError: true
			};
		}

		const note = reply.error
			? `The sub-agent failed: ${reply.error}`
			: reply.stopped
				? 'The sub-agent was stopped before it finished.'
				: !reply.answer
					? 'The sub-agent ended without an answer.'
					: undefined;

		return {
			content: JSON.stringify({
				answer: reply.answer ? truncateSubagentAnswer(reply.answer) : null,
				conversation_id: chat.id,
				note,
				title
			}),
			isError: Boolean(reply.error)
		};
	}

	/**
	 * Waits up to `seconds`, ending early once the user queues a message.
	 * Throws an abort when the flow stops meanwhile.
	 */
	private async sleep(
		conversationId: string,
		seconds: number,
		signal?: AbortSignal
	): Promise<{ slept: number; woken: boolean }> {
		const start = performance.now();
		const end = start + seconds * 1000;
		const elapsed = () => Math.round((performance.now() - start) / 1000);

		while (true) {
			if (signal?.aborted) throw new DOMException('Stopped', 'AbortError');

			if (this.gates.hasPendingSteeringMessage(conversationId)) {
				return { slept: elapsed(), woken: true };
			}

			const left = end - performance.now();

			if (left <= 0) return { slept: elapsed(), woken: false };

			await new Promise((resolve) =>
				setTimeout(resolve, Math.min(left, AGENT_TOOLS.SLEEP_POLL_MS))
			);
		}
	}

	/**
	 * The tools a flow under this policy offers the model, none when agentic
	 * is off. Server tools are fetched first: a disabled /tools endpoint stays
	 * disabled for the life of the server, so the tools panel is the only
	 * place that probes it again. Every settings-enabled MCP server is
	 * initialized, and the collection filters by the policy, so switching
	 * policies never re-initializes connections; a caller without a policy
	 * falls back to the global defaults.
	 */
	private async toolsFor(toolPolicy?: AgenticToolPolicy): Promise<OpenAIToolDefinition[]> {
		if (
			toolsStore.serverTools.length === 0 &&
			!toolsStore.loading &&
			!toolsStore.isToolsEndpointUnreachable
		) {
			await toolsStore.fetchServerTools();
		}

		if (!this.getConfig(settingsStore.config).enabled) return [];

		const disabledTools = new Set(toolPolicy?.disabledTools ?? toolsStore.disabledTools);
		const disabledToolCategories = new Set(
			toolPolicy?.disabledToolCategories ?? toolsStore.disabledToolCategories
		);

		if (conversationsStore.preferences.policyEnabledServerIds().length > 0) {
			const initialized = await mcpStore.ensureInitialized();

			if (!initialized) {
				console.log('[AgenticStore] MCP not initialized');
			}
		}

		return toolsStore.getEnabledToolsForLLM(disabledTools, disabledToolCategories);
	}

	private updateSession(conversationId: string, update: Partial<AgenticSession>): void {
		const session = this.getSession(conversationId);

		this.sessions.set(conversationId, { ...session, ...update });
	}
}

export const agenticStore = new AgenticStore();
