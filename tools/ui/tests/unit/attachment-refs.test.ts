import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types/database';
import {
	assignImageRefIds,
	collectAttachmentRefs,
	formatAttachmentRefLabel,
	formatToolImageRefLabel,
	resolveAttachmentRefs,
	withAttachmentRefHints,
	withFreshImageRefIds,
	withToolImageHandles
} from '$lib/utils/attachment-refs';
import { describe, expect, it } from 'vitest';

const REF = '0f8c2a52-6c1e-4f53-9a57-3b8f1e2d4c6a';
const DATA_URL = 'data:image/png;base64,iVBORw0KGgo=';
const RAW = 'iVBORw0KGgo=';
const refs = new Map([[REF, DATA_URL]]);
const handle = `attachment://${REF}`;

describe('resolveAttachmentRefs', () => {
	it('returns arguments without a handle untouched', () => {
		const args = '{"path":"photo.png"}';

		expect(resolveAttachmentRefs(args, undefined, refs)).toBe(args);
	});

	it('sends a data URI when the schema says nothing about base64', () => {
		const schema = { properties: { image: { type: 'string' } }, type: 'object' };

		expect(resolveAttachmentRefs(`{"image":"${handle}"}`, schema, refs)).toEqual({
			image: DATA_URL
		});
	});

	it('sends raw base64 when the schema asks for it', () => {
		const schema = {
			properties: {
				a: { contentEncoding: 'base64', type: 'string' },
				b: { description: 'Base64 encoded PNG', type: 'string' },
				d: { description: 'Image as a base64 data URI', type: 'string' },
				image_base64: { type: 'string' }
			},
			type: 'object'
		};
		const args = { a: handle, b: handle, d: handle, image_base64: handle };

		expect(resolveAttachmentRefs(args, schema, refs)).toEqual({
			a: RAW,
			b: RAW,
			d: DATA_URL,
			image_base64: RAW
		});
	});

	it('follows arrays and nested objects through the schema', () => {
		const schema = {
			properties: {
				images_base64: { items: { type: 'string' }, type: 'array' },
				meta: {
					properties: { thumb: { description: 'base64 bytes' } },
					type: 'object'
				}
			},
			type: 'object'
		};
		const args = { images_base64: [handle, 'keep'], meta: { other: handle, thumb: handle } };

		expect(resolveAttachmentRefs(args, schema, refs)).toEqual({
			images_base64: [RAW, 'keep'],
			meta: { other: DATA_URL, thumb: RAW }
		});
	});

	it('rejects a handle that names no chat image', () => {
		expect(() =>
			resolveAttachmentRefs('{"image":"attachment://deadbeef"}', undefined, refs)
		).toThrow(/Unknown image handle/);
	});
});

describe('collectAttachmentRefs', () => {
	it('keeps only images that carry a ref id', () => {
		const collected = collectAttachmentRefs([
			{ content: 'api message', role: 'user' },
			{
				extra: [
					{ base64Url: DATA_URL, name: 'a.png', refId: REF, type: AttachmentType.IMAGE },
					{ base64Url: 'data:image/png;base64,AA==', name: 'old.png', type: AttachmentType.IMAGE },
					{ content: 'x', name: 'a.txt', type: AttachmentType.TEXT }
				]
			}
		]);

		expect([...collected.entries()]).toEqual([[REF, DATA_URL]]);
	});
});

describe('formatAttachmentRefLabel', () => {
	it('names the handle', () => {
		expect(formatAttachmentRefLabel(REF)).toContain(handle);
	});
});

describe('formatToolImageRefLabel', () => {
	it('names the handle and marks the image as tool output, not an upload', () => {
		const label = formatToolImageRefLabel(REF);

		expect(label).toContain(handle);
		expect(label).toContain('returned by this tool call, not sent by the user');
		expect(label).not.toBe(formatAttachmentRefLabel(REF));
	});
});

describe('withAttachmentRefHints', () => {
	const schema = {
		properties: {
			images: { description: 'Images to edit', items: { type: 'string' }, type: 'array' },
			meta: { properties: { photo: { type: 'string' } }, type: 'object' },
			prompt: { description: 'What to change in image 1', type: 'string' },
			return_image: { enum: ['preview', 'full'], type: 'string' },
			size: { description: 'Image width', type: 'integer' }
		},
		type: 'object'
	};
	const hinted = withAttachmentRefHints(schema) as {
		properties: Record<
			string,
			{ description?: string; properties?: Record<string, { description?: string }> }
		>;
	};

	it('notes the handle on string parameters that take an image', () => {
		expect(hinted.properties.images.description).toMatch(/^Images to edit .*attachment:\/\/<id>/);
		expect(hinted.properties.meta.properties?.photo.description).toContain('attachment://<id>');
	});

	it('leaves other parameters alone', () => {
		expect(hinted.properties.prompt).toEqual(schema.properties.prompt);
		expect(hinted.properties.return_image).toEqual(schema.properties.return_image);
		expect(hinted.properties.size).toEqual(schema.properties.size);
	});

	it('notes the handle on the start and end frames of a video tool', () => {
		const video = withAttachmentRefHints({
			properties: {
				first_frame: { description: 'Image the video starts from', type: 'string' },
				frame: { description: 'Which frame: "first", "last" or a number', type: 'string' },
				last_frame: { type: 'string' },
				video_frames: { type: 'integer' }
			},
			type: 'object'
		}) as { properties: Record<string, { description?: string }> };

		expect(video.properties.first_frame.description).toMatch(
			/^Image the video starts from .*attachment:\/\/<id>/
		);
		expect(video.properties.last_frame.description).toContain('attachment://<id>');
		expect(video.properties.frame.description).not.toContain('attachment://');
		expect(video.properties.video_frames.description).toBeUndefined();
	});

	it('does not change the encoding a handle is sent as', () => {
		const tool = {
			properties: {
				image: { description: 'Image bytes', type: 'string' },
				image_b64: { description: 'base64 image', type: 'string' }
			},
			type: 'object'
		};
		const args = { image: handle, image_b64: handle };

		expect(resolveAttachmentRefs(args, withAttachmentRefHints(tool), refs)).toEqual(
			resolveAttachmentRefs(args, tool, refs)
		);
		expect(resolveAttachmentRefs(args, withAttachmentRefHints(tool), refs)).toEqual({
			image: DATA_URL,
			image_b64: RAW
		});
	});
});

function counter(): () => string {
	let n = 0;

	return () => `new-${++n}`;
}

const image = (name: string, refId?: string): DatabaseMessageExtra => ({
	base64Url: DATA_URL,
	name,
	refId,
	type: AttachmentType.IMAGE
});
const pdf = (imageRefIds?: string[]): DatabaseMessageExtra => ({
	base64Data: '',
	content: '',
	imageRefIds,
	images: ['data:image/png;base64,AAAA', 'data:image/png;base64,BBBB'],
	name: 'doc.pdf',
	processedAsImages: true,
	type: AttachmentType.PDF
});

describe('collectAttachmentRefs with PDF pages', () => {
	it('maps each page ref id to its page image', () => {
		const found = collectAttachmentRefs([{ extra: [pdf(['p1', 'p2'])] }]);

		expect(found.get('p1')).toBe('data:image/png;base64,AAAA');
		expect(found.get('p2')).toBe('data:image/png;base64,BBBB');
	});
});

describe('assignImageRefIds', () => {
	it('leaves a path whose images all hold distinct ids alone', () => {
		const path = [
			{ extra: [image('a.png', 'r1')], id: 'm1' },
			{ extra: [pdf(['p1', 'p2'])], id: 'm2' }
		];

		expect(assignImageRefIds(path, counter()).size).toBe(0);
	});

	it('gives images without an id one', () => {
		const changed = assignImageRefIds(
			[{ extra: [image('a.png'), image('b.png', 'r2')], id: 'm1' }],
			counter()
		);

		expect(changed.get('m1')).toEqual([image('a.png', 'new-1'), image('b.png', 'r2')]);
	});

	it('gives a later copy of an image a new id and keeps the first', () => {
		const changed = assignImageRefIds(
			[
				{ extra: [image('a.png', 'r1')], id: 'm1' },
				{ extra: [], id: 'm2' },
				{ extra: [image('a.png', 'r1')], id: 'm3' }
			],
			counter()
		);

		expect([...changed.keys()]).toEqual(['m3']);
		expect(changed.get('m3')).toEqual([image('a.png', 'new-1')]);
	});

	it('fills and dedupes PDF page ids', () => {
		const changed = assignImageRefIds(
			[
				{ extra: [image('a.png', 'r1')], id: 'm1' },
				{ extra: [pdf(['r1'])], id: 'm2' }
			],
			counter()
		);

		expect(changed.get('m2')).toEqual([pdf(['new-1', 'new-2'])]);
	});
});

describe('withFreshImageRefIds', () => {
	it('renews image and PDF page ids and keeps other extras', () => {
		const text: DatabaseMessageExtra = { content: 'x', name: 'a.txt', type: AttachmentType.TEXT };

		expect(
			withFreshImageRefIds([image('a.png', 'r1'), pdf(['p1', 'p2']), text], counter())
		).toEqual([image('a.png', 'new-1'), pdf(['new-2', 'new-3']), text]);
		expect(withFreshImageRefIds(undefined)).toBeUndefined();
	});
});

describe('withToolImageHandles', () => {
	it('labels each saved image line with its handle', () => {
		const content = 'map ready\n[Attachment saved: a.png]\n[Attachment saved: b.png]\ndone';

		expect(withToolImageHandles(content, [image('a.png', 'r1'), image('b.png', 'r2')])).toBe(
			`map ready\n[Attachment saved: a.png] ${formatToolImageRefLabel('r1')}\n` +
				`[Attachment saved: b.png] ${formatToolImageRefLabel('r2')}\ndone`
		);
	});

	it('returns text without image ids untouched', () => {
		const content = 'x\n[Attachment saved: a.png]';

		expect(withToolImageHandles(content, [image('a.png')])).toBe(content);
		expect(withToolImageHandles(content, undefined)).toBe(content);
	});

	it('labels an image without a saved line at the end, and a doubly stored image once', () => {
		const content = '[Attachment saved: a.png]';

		expect(
			withToolImageHandles(content, [
				image('a.png', 'r1'),
				image('a.png', 'r9'),
				image('c.png', 'r3')
			])
		).toBe(
			`[Attachment saved: a.png] ${formatToolImageRefLabel('r1')}\n${formatToolImageRefLabel('r3')}`
		);
	});

	it('keeps the labels in place when a reminder is appended later', () => {
		const content = '[Attachment saved: a.png]';
		const extras = [image('a.png', 'r1')];

		expect(withToolImageHandles(`${content}\n\nreminder`, extras)).toBe(
			`${withToolImageHandles(content, extras)}\n\nreminder`
		);
	});
});
