/**
 * Read-aloud text is spoken in chunks: Chrome cuts off one long utterance
 * after about 15 seconds.
 */
export const SPEECH_CHUNK_MAX_CHARS = 200;

/** Speech recognition errors that come from a blocked microphone. */
export const SPEECH_RECOGNITION_BLOCKED_ERRORS = ['not-allowed', 'service-not-allowed'];

/** Speech recognition errors that need no message to the user. */
export const SPEECH_RECOGNITION_SILENT_ERRORS = ['aborted', 'no-speech'];

export const SPEECH_MIC_BLOCKED_MESSAGE =
	'Microphone blocked. Dictation needs microphone permission and an HTTPS or localhost address.';

/** Dictation sent to a transcription server is 16 kHz WAV, the rate whisper models use. */
export const TRANSCRIPTION_SAMPLE_RATE = 16000;
