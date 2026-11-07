import { TranscriptionService } from '$lib/services/transcription.service';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/utils/api-headers', () => ({
	getAuthHeaders: () => ({ Authorization: 'Bearer secret' })
}));

function mockFetch(response: Response) {
	const fetchMock = vi.fn().mockResolvedValue(response);

	vi.stubGlobal('fetch', fetchMock);

	return fetchMock;
}

const audio = new File([new Uint8Array([1, 2, 3])], 'dictation.wav', { type: 'audio/wav' });

describe('TranscriptionService.transcribe', () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('posts the audio to this server with the API key', async () => {
		const fetchMock = mockFetch(Response.json({ text: '  hello \n world ' }));
		const text = await TranscriptionService.transcribe(audio, { language: 'en', model: 'asr' });
		const [url, init] = fetchMock.mock.calls[0];
		const form = init.body as FormData;

		expect(text).toBe('hello world');
		expect(String(url)).toMatch(/\/v1\/audio\/transcriptions$/);
		expect(init.headers).toEqual({ Authorization: 'Bearer secret' });
		expect(form.get('file')).toBeInstanceOf(File);
		expect(form.get('response_format')).toBe('json');
		expect(form.get('model')).toBe('asr');
		expect(form.get('language')).toBe('en');
	});

	it('does not send the API key to another server', async () => {
		const fetchMock = mockFetch(Response.json({ text: 'hi' }));

		await TranscriptionService.transcribe(audio, { url: ' http://whisper:8080/inference ' });

		const [url, init] = fetchMock.mock.calls[0];

		expect(url).toBe('http://whisper:8080/inference');
		expect(init.headers).toEqual({});
		expect((init.body as FormData).has('model')).toBe(false);
	});

	it('throws the server error message', async () => {
		mockFetch(
			Response.json(
				{ error: { message: 'The current model does not support audio input.' } },
				{ status: 501 }
			)
		);

		await expect(TranscriptionService.transcribe(audio)).rejects.toThrow(
			'The current model does not support audio input.'
		);
	});

	it('names the URL when the server cannot be reached', async () => {
		vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

		await expect(
			TranscriptionService.transcribe(audio, { url: 'http://whisper:8080/inference' })
		).rejects.toThrow('Cannot reach http://whisper:8080/inference');
	});
});
