import { base } from '$app/paths';
import { API_AUDIO } from '$lib/constants';
import { parseErrorMessage } from '$lib/utils/api-fetch';
import { getAuthHeaders } from '$lib/utils/api-headers';

export interface TranscriptionOptions {
	/** Endpoint URL; empty means this server. */
	url?: string;
	model?: string;
	/** ISO-639-1 code, e.g. "en". */
	language?: string;
	signal?: AbortSignal;
}

/**
 * Speech to text over the OpenAI transcriptions format (multipart form,
 * audio in the file field, JSON reply with a text field). llama-server,
 * whisper.cpp server and OpenAI-style whisper servers all answer it.
 */
export class TranscriptionService {
	static async transcribe(audio: File, options: TranscriptionOptions = {}): Promise<string> {
		const customUrl = options.url?.trim();
		const url = customUrl || `${base}${API_AUDIO.TRANSCRIPTIONS}`;
		const form = new FormData();

		form.append('file', audio);
		form.append('response_format', 'json');

		if (options.model) form.append('model', options.model);

		if (options.language) form.append('language', options.language);

		let response: Response;

		try {
			// the API key belongs to this server, never send it to another one
			response = await fetch(url, {
				body: form,
				headers: customUrl ? {} : getAuthHeaders(),
				method: 'POST',
				signal: options.signal
			});
		} catch (error) {
			if (error instanceof Error && error.name === 'AbortError') throw error;

			throw new Error(`Cannot reach ${url} (offline, or it does not allow CORS)`);
		}

		if (!response.ok) {
			throw new Error(await parseErrorMessage(response));
		}

		const data = await response.json();

		return typeof data?.text === 'string' ? data.text.replace(/\s+/g, ' ').trim() : '';
	}
}
