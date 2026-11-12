// Videos a tool links (sd-server generate_video) play under the tool block,
// so they are visible without expanding it.

import ChatMessageToolCallBlockDefault from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockDefault.svelte';
import { AgenticSectionType, AttachmentType } from '$lib/enums';
import type { AgenticSection, DatabaseMessageExtra } from '$lib/types';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const VIDEO_NAME = 'img-1a2b-3.webm';
const section: AgenticSection = {
	content: '',
	toolArgs: '{"prompt":"a cat"}',
	toolName: 'generate_video',
	toolResult: '{"videos":[{"url":"http://127.0.0.1:8080/mcp/images/img-1a2b-3.webm"}]}',
	type: AgenticSectionType.TOOL_CALL
};
const video: DatabaseMessageExtra = {
	base64Data: 'GkXfow==',
	mimeType: 'video/webm',
	name: VIDEO_NAME,
	type: AttachmentType.VIDEO
};

describe('tool call videos', () => {
	it('shows a player while the block is collapsed', async () => {
		render(ChatMessageToolCallBlockDefault, {
			attachments: [video],
			isStreaming: false,
			open: false,
			section
		});

		await tick();

		const players = document.querySelectorAll('video');

		expect(players).toHaveLength(1);
		expect(players[0].controls).toBe(true);
		expect(players[0].getAttribute('aria-label')).toBe(VIDEO_NAME);
		expect(players[0].getAttribute('src')).toBe('data:video/webm;base64,GkXfow==');
	});

	it('shows a video stored twice only once', async () => {
		render(ChatMessageToolCallBlockDefault, {
			attachments: [video, video],
			isStreaming: false,
			open: true,
			section
		});

		await tick();

		expect(document.querySelectorAll('video')).toHaveLength(1);
	});
});
