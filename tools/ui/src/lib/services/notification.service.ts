/**
 * NotificationService - Notifications the model sends with the notify tool
 *
 * Shows a system notification when the browser allows one: through the
 * service worker where it runs (the only way on mobile), else with the
 * Notification constructor. Without permission, or on a page served over
 * plain HTTP where the API does not exist, it falls back to an in-page toast
 * and marks the tab title until the page is looked at again.
 */

import { toast } from 'svelte-sonner';

/** Where a notification was shown. */
export type NotificationOutcome = 'system' | 'page';

const TITLE_MARK = '(!) ';

let titleMarked = false;

function markTitleUntilVisible(): void {
	if (titleMarked || document.visibilityState === 'visible') return;

	titleMarked = true;
	document.title = TITLE_MARK + document.title;

	const clear = () => {
		if (document.visibilityState !== 'visible') return;

		document.removeEventListener('visibilitychange', clear);
		titleMarked = false;

		if (document.title.startsWith(TITLE_MARK))
			document.title = document.title.slice(TITLE_MARK.length);
	};

	document.addEventListener('visibilitychange', clear);
}

export class NotificationService {
	/** Whether the page may show system notifications, asking once when it has not decided. */
	static async ensurePermission(): Promise<boolean> {
		if (typeof Notification === 'undefined') return false;

		if (Notification.permission === 'granted') return true;

		if (Notification.permission === 'denied') return false;

		try {
			return (await Notification.requestPermission()) === 'granted';
		} catch {
			// browsers that only ask from a click reject here
			return false;
		}
	}

	static async show(title: string, message: string): Promise<NotificationOutcome> {
		if (await NotificationService.ensurePermission()) {
			try {
				const registration = await navigator.serviceWorker?.getRegistration();

				if (registration) {
					await registration.showNotification(title, { body: message, tag: 'llama-notify' });
				} else {
					new Notification(title, { body: message, tag: 'llama-notify' });
				}

				return 'system';
			} catch (error) {
				console.warn('[NotificationService] System notification failed:', error);
			}
		}

		toast.info(title, { description: message, duration: 15000 });
		markTitleUntilVisible();

		return 'page';
	}
}
