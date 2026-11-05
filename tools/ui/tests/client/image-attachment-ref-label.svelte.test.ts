// A chat image with a ref id goes to the model with a text label naming its
// attachment:// handle, so the model can pass the image to a tool.

import { AttachmentType, MessageRole } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import type { DatabaseMessage } from '$lib/types';
import { describe, expect, it } from 'vitest';

const PNG =
	'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
const REF = '0f8c2a52-6c1e-4f53-9a57-3b8f1e2d4c6a';

function userMessage(refId?: string): DatabaseMessage {
	return {
		children: [],
		content: 'upload this',
		convId: 'c',
		extra: [{ base64Url: PNG, name: 'photo.png', refId, type: AttachmentType.IMAGE }],
		id: 'm',
		parent: null,
		role: MessageRole.USER,
		timestamp: 0,
		type: 'text'
	};
}

describe('image attachment ref label', () => {
	it('puts the handle label right before the image', async () => {
		const api = await ChatService.convertDbMessageToApiChatMessageData(userMessage(REF));

		expect(Array.isArray(api.content)).toBe(true);

		const parts = api.content as { type: string; text?: string }[];

		expect(parts.map((p) => p.type)).toEqual(['text', 'image_url', 'text']);
		expect(parts[0].text).toContain(`attachment://${REF}`);
		expect(parts[2].text).toBe('upload this');
	});

	it('adds no label to images sent before ref ids existed', async () => {
		const api = await ChatService.convertDbMessageToApiChatMessageData(userMessage());
		const parts = api.content as { type: string }[];

		expect(parts.map((p) => p.type)).toEqual(['image_url', 'text']);
	});
});
