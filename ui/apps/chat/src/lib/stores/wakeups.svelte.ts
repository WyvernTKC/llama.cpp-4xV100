/**
 * wakeupsStore - Scheduled wake-ups of conversations
 *
 * Holds the wake-ups the model schedules with schedule_wakeup, one per
 * conversation, and fires them: once a wake-up is due, the registered handler
 * (chatStore) sends its note into the conversation as a message. A chat that
 * is still busy is tried again a little later. Wake-ups persist in
 * localStorage, so a reload keeps them; they fire only while the page is open.
 */

import { AGENT_TOOLS } from '#lib/constants/index.ts';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import { browser } from '$app/env';
import { SvelteMap } from 'svelte/reactivity';

export interface ScheduledWakeup {
	convId: string;
	/** When it fires, in ms since the epoch. */
	dueAt: number;
	/** What the model asked to be told when it wakes up. */
	note: string;
}

/** Sends a due wake-up; resolves false when the chat is busy and it should try again. */
type WakeupHandler = (convId: string, note: string) => Promise<boolean>;

function isWakeup(value: unknown): value is ScheduledWakeup {
	if (!value || typeof value !== 'object') return false;

	const w = value as Record<string, unknown>;

	return (
		typeof w.convId === 'string' &&
		typeof w.dueAt === 'number' &&
		Number.isFinite(w.dueAt) &&
		typeof w.note === 'string'
	);
}

class WakeupsStore {
	private handler: WakeupHandler | null = null;
	private timers = new Map<string, ReturnType<typeof setTimeout>>();
	private wakeups = new SvelteMap<string, ScheduledWakeup>();

	/** Drop the wake-up of a conversation; false when it had none. */
	cancel(convId: string): boolean {
		clearTimeout(this.timers.get(convId));
		this.timers.delete(convId);

		const had = this.wakeups.delete(convId);

		if (had) this.persist();

		return had;
	}

	constructor() {
		conversationsStore.onConversationsDeleted((convIds) => {
			for (const convId of convIds) this.cancel(convId);
		});
	}

	get(convId: string): ScheduledWakeup | null {
		return this.wakeups.get(convId) ?? null;
	}

	/**
	 * Register the sender of due wake-ups and arm the stored ones. Called once
	 * by chatStore; wake-ups loaded before that wait for it.
	 */
	initialize(handler: WakeupHandler): void {
		this.handler = handler;

		if (!browser) return;

		try {
			const raw = localStorage.getItem(AGENT_TOOLS.WAKEUP_STORAGE_KEY);
			const parsed: unknown = raw ? JSON.parse(raw) : [];

			if (Array.isArray(parsed)) {
				for (const item of parsed) {
					if (isWakeup(item)) this.wakeups.set(item.convId, item);
				}
			}
		} catch {
			// a broken entry only loses the stored wake-ups
		}

		// one that fell due while the page was closed waits until the app has loaded
		const earliest = Date.now() + AGENT_TOOLS.WAKEUP_STARTUP_DELAY_MS;

		for (const wakeup of this.wakeups.values()) {
			this.arm(wakeup.convId, Math.max(wakeup.dueAt, earliest));
		}
	}

	/** Schedule the wake-up of a conversation, replacing the one it had. */
	schedule(convId: string, minutes: number, note: string): ScheduledWakeup {
		const wakeup: ScheduledWakeup = { convId, dueAt: Date.now() + minutes * 60_000, note };

		this.wakeups.set(convId, wakeup);
		this.persist();
		this.arm(convId, wakeup.dueAt);

		return wakeup;
	}

	private arm(convId: string, at: number): void {
		clearTimeout(this.timers.get(convId));
		this.timers.set(
			convId,
			setTimeout(() => void this.fire(convId), Math.max(0, at - Date.now()))
		);
	}

	private async fire(convId: string): Promise<void> {
		this.timers.delete(convId);

		const wakeup = this.wakeups.get(convId);

		if (!wakeup || !this.handler) return;

		let sent: boolean;

		try {
			sent = await this.handler(convId, wakeup.note);
		} catch (error) {
			console.error('[WakeupsStore] Failed to send a wake-up:', error);
			// a chat that cannot take the message now will not later either
			sent = true;
		}

		// the handler may have run long; a wake-up scheduled meanwhile replaces this one
		if (this.wakeups.get(convId) !== wakeup) return;

		if (sent) {
			this.wakeups.delete(convId);
			this.persist();

			return;
		}

		this.arm(convId, Date.now() + AGENT_TOOLS.WAKEUP_RETRY_MS);
	}

	private persist(): void {
		if (!browser) return;

		try {
			localStorage.setItem(
				AGENT_TOOLS.WAKEUP_STORAGE_KEY,
				JSON.stringify([...this.wakeups.values()])
			);
		} catch {
			// storage full or blocked: the wake-ups stay for this page only
		}
	}
}

export const wakeupsStore = new WakeupsStore();
