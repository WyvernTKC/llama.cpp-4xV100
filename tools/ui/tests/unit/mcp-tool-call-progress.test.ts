import { parseMcpToolCallProgress } from '$lib/utils/mcp';
import { describe, expect, it } from 'vitest';

// the _meta key and field names are the wire contract with the stable-diffusion.cpp server
function withPreview(preview: Record<string, unknown>) {
	return { _meta: { 'sdcpp/progress': { percent: 40, phase: 'sampling', preview } } };
}

describe('parseMcpToolCallProgress', () => {
	it('keeps progress, total and message from a plain notification', () => {
		expect(
			parseMcpToolCallProgress({
				message: 'sampling step 7/20, about 38 s left',
				progress: 31.5,
				total: 100
			})
		).toEqual({
			message: 'sampling step 7/20, about 38 s left',
			previewUrl: undefined,
			progress: 31.5,
			total: 100
		});
	});

	it('turns a preview into a data URL and drops whitespace from its base64', () => {
		const progress = parseMcpToolCallProgress({
			progress: 40,
			total: 100,
			...withPreview({ data: 'iVBORw0K\nGgo=', mime_type: 'image/png', revision: 7 })
		});

		expect(progress.previewUrl).toBe('data:image/png;base64,iVBORw0KGgo=');
	});

	it('keeps the previous preview until a newer one arrives', () => {
		const first = parseMcpToolCallProgress({
			progress: 40,
			total: 100,
			...withPreview({ data: 'AAAA', mime_type: 'image/png', revision: 7 })
		});
		const second = parseMcpToolCallProgress(
			{ message: 'step 8/20', progress: 45, total: 100 },
			first
		);
		const third = parseMcpToolCallProgress(
			{
				progress: 50,
				total: 100,
				...withPreview({ data: 'BBBB', mime_type: 'image/webp', revision: 9 })
			},
			second
		);

		expect(second.previewUrl).toBe('data:image/png;base64,AAAA');
		expect(second.message).toBe('step 8/20');
		expect(third.previewUrl).toBe('data:image/webp;base64,BBBB');
	});

	it('ignores previews with a non-raster type, a missing field or too much data', () => {
		const previous = parseMcpToolCallProgress({
			progress: 10,
			...withPreview({ data: 'AAAA', mime_type: 'image/jpeg' })
		});

		for (const preview of [
			{ data: 'PHN2Zz4=', mime_type: 'image/svg+xml' },
			{ data: 'AAAA' },
			{ data: 42, mime_type: 'image/png' },
			{ data: 'A'.repeat(1_000_001), mime_type: 'image/png' }
		]) {
			expect(
				parseMcpToolCallProgress({ progress: 20, ...withPreview(preview) }, previous).previewUrl
			).toBe('data:image/jpeg;base64,AAAA');
		}
	});

	it('ignores a malformed or foreign _meta', () => {
		for (const _meta of [
			{},
			{ 'sdcpp/progress': 'x' },
			{ 'sdcpp/progress': null },
			{ other: { preview: {} } }
		]) {
			expect(parseMcpToolCallProgress({ _meta, progress: 5 }).previewUrl).toBeUndefined();
		}
	});
});
