// Images a tool returns render under the tool block, so they are visible
// without expanding it, and a tap opens them full size with a Save button.

import ChatMessageToolCallBlockDefault from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockDefault.svelte';
import { AgenticSectionType, AttachmentType } from '$lib/enums';
import type { AgenticSection, DatabaseMessageExtra } from '$lib/types';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

const IMAGE_NAME = 'mcp-attachment-1-1.png';
const IMAGE_URL =
	'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
const section: AgenticSection = {
	content: '',
	toolArgs: '{"city":"Oslo"}',
	toolName: 'static_map',
	toolResult: `map ready\n[Attachment saved: ${IMAGE_NAME}]`,
	type: AgenticSectionType.TOOL_CALL
};
const attachments: DatabaseMessageExtra[] = [
	{ base64Url: IMAGE_URL, name: IMAGE_NAME, type: AttachmentType.IMAGE }
];

describe('tool call images', () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('shows the image while the block is collapsed', async () => {
		render(ChatMessageToolCallBlockDefault, {
			attachments,
			isStreaming: false,
			open: false,
			section
		});

		await tick();

		const images = document.querySelectorAll('img');

		expect(images).toHaveLength(1);
		expect(images[0].getAttribute('src')).toBe(IMAGE_URL);
		expect(document.body.textContent).not.toContain('map ready');
	});

	it('shows the image once when the block is open', async () => {
		render(ChatMessageToolCallBlockDefault, {
			attachments,
			isStreaming: false,
			open: true,
			section
		});

		await tick();

		expect(document.body.textContent).toContain('map ready');
		expect(document.querySelectorAll('img')).toHaveLength(1);
	});

	it('shows an image stored twice by older chats only once', async () => {
		render(ChatMessageToolCallBlockDefault, {
			attachments: [...attachments, ...attachments],
			isStreaming: false,
			open: false,
			section
		});

		await tick();

		expect(document.querySelectorAll('img')).toHaveLength(1);
	});

	it('opens the image on tap and saves it under its name', async () => {
		const saved: { download: string; href: string }[] = [];

		vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (
			this: HTMLAnchorElement
		) {
			saved.push({ download: this.download, href: this.href });
		});

		render(ChatMessageToolCallBlockDefault, {
			attachments,
			isStreaming: false,
			open: false,
			section
		});

		await tick();
		document.querySelector<HTMLButtonElement>(`button[aria-label="Open ${IMAGE_NAME}"]`)!.click();

		const saveButton = await vi.waitFor(() => {
			const button = [...document.querySelectorAll('button')].find(
				(b) => b.textContent?.trim() === 'Save'
			);

			expect(button).toBeDefined();

			return button!;
		});

		expect(
			[...document.querySelectorAll('img')].filter((img) => img.src === IMAGE_URL).length
		).toBeGreaterThan(1);

		saveButton.click();

		await vi.waitFor(() => expect(saved).toHaveLength(1));
		expect(saved[0].download).toBe(IMAGE_NAME);
		expect(saved[0].href).toMatch(/^blob:/);
	});
});
