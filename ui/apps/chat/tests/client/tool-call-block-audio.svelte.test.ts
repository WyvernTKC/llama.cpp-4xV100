// Audio a tool returns (a music server's generate_song) plays under the tool
// block, so it is visible without expanding it.

import ChatMessageToolCallBlockDefault from '#lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockDefault.svelte';
import { AgenticSectionType, AttachmentType } from '#lib/enums/index.ts';
import type { AgenticSection, DatabaseMessageExtra } from '#lib/types/index.ts';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const AUDIO_NAME = 'mcp-attachment-1700000000000-1.wav';
const section: AgenticSection = {
	content: '',
	toolArgs: '{"caption":"lo-fi","seconds":10}',
	toolName: 'generate_song',
	toolResult: `{\n  "id": "song-1",\n  "seconds": 10.0\n}\n[Attachment saved: ${AUDIO_NAME}]`,
	type: AgenticSectionType.TOOL_CALL
};
const audio: DatabaseMessageExtra = {
	base64Data: 'UklGRiQAAABXQVZF',
	mimeType: 'audio/wav',
	name: AUDIO_NAME,
	type: AttachmentType.AUDIO
};

describe('tool call audio', () => {
	it('shows a player while the block is collapsed', async () => {
		render(ChatMessageToolCallBlockDefault, {
			attachments: [audio],
			isStreaming: false,
			open: false,
			section
		});

		await tick();

		const players = document.querySelectorAll('audio');

		expect(players).toHaveLength(1);
		expect(players[0].controls).toBe(true);
		expect(players[0].getAttribute('aria-label')).toBe(AUDIO_NAME);
		expect(players[0].getAttribute('src')).toBe('data:audio/wav;base64,UklGRiQAAABXQVZF');
	});

	it('shows one player when the block is open', async () => {
		render(ChatMessageToolCallBlockDefault, {
			attachments: [audio, audio],
			isStreaming: false,
			open: true,
			section
		});

		await tick();

		expect(document.querySelectorAll('audio')).toHaveLength(1);
	});
});
