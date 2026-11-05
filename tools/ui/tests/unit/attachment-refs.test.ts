import { AttachmentType } from '$lib/enums';
import {
	collectAttachmentRefs,
	formatAttachmentRefLabel,
	resolveAttachmentRefs
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
