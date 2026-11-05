/**
 * speechStore - Read-aloud of chat replies
 *
 * Speaks message text with the device voice (Web Speech API
 * speechSynthesis), one message at a time. speakingId names the message
 * being read, so its read-aloud button can show a stop state.
 */

import {
	isSpeechSynthesisSupported,
	markdownToSpeechText,
	splitSpeechText
} from '$lib/utils/speech';

class SpeechStore {
	/** Id of the message being read aloud, null when silent. */
	speakingId = $state<string | null>(null);

	private unlocked = false;
	// keep the queued utterances referenced: Chrome drops the events of collected ones
	private utterances: SpeechSynthesisUtterance[] = [];

	get isSupported(): boolean {
		return isSpeechSynthesisSupported();
	}

	speak(id: string, markdown: string): void {
		this.stop();

		if (!isSpeechSynthesisSupported()) return;

		const chunks = splitSpeechText(markdownToSpeechText(markdown));

		if (chunks.length === 0) return;

		const utterances = chunks.map((chunk) => {
			const utterance = new SpeechSynthesisUtterance(chunk);

			utterance.lang = navigator.language;

			return utterance;
		});
		const last = utterances[utterances.length - 1];

		last.onend = last.onerror = () => {
			if (this.utterances !== utterances) return;

			this.utterances = [];
			this.speakingId = null;
		};

		this.utterances = utterances;
		this.speakingId = id;

		for (const utterance of utterances) {
			speechSynthesis.speak(utterance);
		}
	}

	stop(): void {
		this.utterances = [];
		this.speakingId = null;

		if (isSpeechSynthesisSupported()) speechSynthesis.cancel();
	}

	toggle(id: string, markdown: string): void {
		if (this.speakingId === id) {
			this.stop();
		} else {
			this.speak(id, markdown);
		}
	}

	/** iOS lets a page speak only after it spoke once inside a tap; call this from a tap handler. */
	unlock(): void {
		if (this.unlocked || !isSpeechSynthesisSupported()) return;

		const utterance = new SpeechSynthesisUtterance('');

		utterance.volume = 0;
		speechSynthesis.speak(utterance);
		this.unlocked = true;
	}
}

export const speechStore = new SpeechStore();
