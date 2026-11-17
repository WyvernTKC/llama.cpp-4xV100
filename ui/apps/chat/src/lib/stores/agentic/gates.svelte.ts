/**
 * AgenticGates - User interaction gates for the agentic loop
 *
 * Owns the state the loop waits on between turns: tool permission requests,
 * turn-limit continue prompts, ask_user questions and queued steering
 * messages. The loop awaits requestPermission/requestContinue/requestAnswer;
 * the UI resolves them through resolvePermission/resolveContinue/
 * resolveQuestion. Owned by agenticStore, no host coupling.
 */

import { ToolPermissionDecision } from '#lib/enums/index.ts';
// direct imports between stores, not via the barrel, to avoid circular deps
import { permissionsStore } from '#lib/stores/permissions.svelte.ts';
import { toolsStore } from '#lib/stores/tools.svelte.ts';
import type { DatabaseMessageExtra, PendingQuestion, SteeringMessage } from '#lib/types/index.ts';
import { SvelteMap } from 'svelte/reactivity';

export class AgenticGates {
	/** Resolve functions for pending continue Promises; nothing derives from this map */
	private continueResolvers = new SvelteMap<string, (shouldContinue: boolean) => void>();
	/** Dedicated reactive state for pending continue requests (turn limit reached) */
	private pendingContinueRequests = new SvelteMap<string, boolean>();

	/** Dedicated reactive state for pending permission requests (ensures immediate UI updates) */
	private pendingPermissions = new SvelteMap<
		string,
		{ toolName: string; serverLabel: string } | null
	>();
	/** Reactive: the ask_user question a conversation waits on */
	private pendingQuestions = new SvelteMap<string, PendingQuestion>();

	/** Resolve functions for pending permission Promises; nothing derives from this map */
	private permissionResolvers = new SvelteMap<string, (decision: ToolPermissionDecision) => void>();
	/** Resolve functions for pending questions, null for a stopped one; nothing derives from this map */
	private questionResolvers = new SvelteMap<string, (answer: string | null) => void>();

	/** Reactive: queued steering messages to inject between turns */
	private steeringMessages = new SvelteMap<string, SteeringMessage>();

	/**
	 * Drop all pending gate state for a conversation, e.g. when a flow exits.
	 */
	clear(conversationId: string): void {
		this.pendingPermissions.set(conversationId, null);
		this.permissionResolvers.delete(conversationId);
		this.pendingContinueRequests.set(conversationId, false);
		this.continueResolvers.delete(conversationId);
		this.steeringMessages.delete(conversationId);
		this.questionResolvers.get(conversationId)?.(null);
	}

	/**
	 * Clear the pending steering message without consuming it.
	 */
	clearSteeringMessage(conversationId: string): void {
		this.steeringMessages.delete(conversationId);
	}

	/**
	 * Consume and return the pending steering message for re-sending.
	 * Called by chatStore after the agentic flow exits.
	 */
	consumePendingSteeringMessage(conversationId: string): SteeringMessage | null {
		const msg = this.steeringMessages.get(conversationId);

		if (!msg) return null;

		this.steeringMessages.delete(conversationId);

		return msg;
	}

	getPendingContinueRequest(conversationId: string): boolean {
		return this.pendingContinueRequests.get(conversationId) ?? false;
	}

	getPendingPermissionRequest(
		conversationId: string
	): { toolName: string; serverLabel: string } | null {
		return this.pendingPermissions.get(conversationId) ?? null;
	}

	getPendingQuestion(conversationId: string): PendingQuestion | null {
		return this.pendingQuestions.get(conversationId) ?? null;
	}

	getPendingSteeringMessageContent(conversationId: string): string | null {
		return this.steeringMessages.get(conversationId)?.content ?? null;
	}

	getPendingSteeringMessageExtras(conversationId: string): DatabaseMessageExtra[] | undefined {
		return this.steeringMessages.get(conversationId)?.extras;
	}

	hasPendingSteeringMessage(conversationId: string): boolean {
		return this.steeringMessages.has(conversationId);
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
		this.steeringMessages.set(conversationId, { content, extras });
	}

	/**
	 * Show an ask_user question and wait for the answer: a picked option or
	 * text the user typed. Null when the flow stops before an answer.
	 */
	async requestAnswer(
		conversationId: string,
		question: PendingQuestion,
		signal?: AbortSignal
	): Promise<string | null> {
		// a question still open from an earlier flow can never be answered now
		this.questionResolvers.get(conversationId)?.(null);

		if (signal?.aborted) return null;

		return new Promise<string | null>((resolve) => {
			const onAbort = () => this.questionResolvers.get(conversationId)?.(null);

			this.pendingQuestions.set(conversationId, question);
			this.questionResolvers.set(conversationId, (answer) => {
				this.questionResolvers.delete(conversationId);
				this.pendingQuestions.delete(conversationId);
				signal?.removeEventListener('abort', onAbort);
				resolve(answer);
			});

			signal?.addEventListener('abort', onAbort, { once: true });
		});
	}

	async requestContinue(conversationId: string, signal?: AbortSignal): Promise<boolean> {
		this.pendingContinueRequests.set(conversationId, true);

		return new Promise<boolean>((resolve) => {
			if (signal?.aborted) {
				this.pendingContinueRequests.set(conversationId, false);
				resolve(false);

				return;
			}

			this.continueResolvers.set(conversationId, (shouldContinue) => {
				this.pendingContinueRequests.set(conversationId, false);
				resolve(shouldContinue);
			});

			signal?.addEventListener(
				'abort',
				() => {
					const resolver = this.continueResolvers.get(conversationId);

					if (resolver) {
						this.continueResolvers.delete(conversationId);
						this.pendingContinueRequests.set(conversationId, false);
						resolve(false);
					}
				},
				{ once: true }
			);
		});
	}

	async requestPermission(
		conversationId: string,
		toolName: string,
		serverLabel: string,
		signal?: AbortSignal
	): Promise<ToolPermissionDecision> {
		const permissionKey = toolsStore.getPermissionKey(toolName);

		if (permissionKey && permissionsStore.hasTool(permissionKey)) {
			return ToolPermissionDecision.ONCE;
		}

		this.pendingPermissions.set(conversationId, { serverLabel, toolName });

		return new Promise<ToolPermissionDecision>((resolve) => {
			if (signal?.aborted) {
				this.pendingPermissions.set(conversationId, null);
				resolve(ToolPermissionDecision.DENY);

				return;
			}

			this.permissionResolvers.set(conversationId, (decision) => {
				this.pendingPermissions.set(conversationId, null);

				if (decision === ToolPermissionDecision.ALWAYS && permissionKey) {
					permissionsStore.allowTool(permissionKey);
				} else if (decision === ToolPermissionDecision.ALWAYS_SERVER) {
					const serverToolKeys = toolsStore.allTools
						.filter((t) =>
							t.serverName
								? t.serverName === serverLabel
								: toolsStore.getToolServerLabel(t.definition.function.name) === serverLabel
						)
						.map((t) => toolsStore.getPermissionKey(t.definition.function.name)!)
						.filter((k): k is string => k !== null);

					permissionsStore.allowTools(serverToolKeys);
				}

				resolve(decision);
			});

			signal?.addEventListener(
				'abort',
				() => {
					const resolver = this.permissionResolvers.get(conversationId);

					if (resolver) {
						this.permissionResolvers.delete(conversationId);
						this.pendingPermissions.set(conversationId, null);
						resolve(ToolPermissionDecision.DENY);
					}
				},
				{ once: true }
			);
		});
	}

	resolveContinue(conversationId: string, shouldContinue: boolean): void {
		const resolver = this.continueResolvers.get(conversationId);

		if (resolver) {
			this.continueResolvers.delete(conversationId);
			resolver(shouldContinue);
		}
	}

	resolvePermission(conversationId: string, decision: ToolPermissionDecision): void {
		const resolver = this.permissionResolvers.get(conversationId);

		if (resolver) {
			this.permissionResolvers.delete(conversationId);
			resolver(decision);
		}
	}

	/** Answer the open ask_user question; false when there is none. */
	resolveQuestion(conversationId: string, answer: string): boolean {
		const resolver = this.questionResolvers.get(conversationId);

		if (!resolver) return false;

		resolver(answer);

		return true;
	}
}
