import { CORS_PROXY, CORS_PROXY_ENDPOINT, MCP_TOOL_VIDEO_MAX_COUNT } from '#lib/constants/index.ts';
import { AttachmentType, MCPTransportType } from '#lib/enums/index.ts';
import { MCPService } from '#lib/services/mcp.service.ts';
import type { MCPServerConfig } from '#lib/types/index.ts';
import { findMcpToolResultVideoUrls, getMcpToolVideoMimeType } from '#lib/utils/index.ts';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const SERVER_URL = 'http://192.168.1.20:8080/mcp';
const VIDEO_URL = 'http://192.168.1.20:8080/mcp/images/img-1a2b-3.webm';
// the text sd-server's generate_video returns, followed by its frame sheet
const SD_SERVER_RESULT = [
	'{',
	'  "job_id": "job_1_2",',
	'  "videos": [',
	'    {',
	'      "fps": 24,',
	'      "has_audio": true,',
	'      "id": "img-1a2b-3",',
	`      "url": "${VIDEO_URL}"`,
	'    }',
	'  ],',
	'  "note": "Give the user the url to watch the video (with sound)."',
	'}',
	'data:image/jpeg;base64,/9j/4AAQ'
].join('\n');
const config = (overrides: Partial<MCPServerConfig> = {}): MCPServerConfig => ({
	transport: MCPTransportType.STREAMABLE_HTTP,
	url: SERVER_URL,
	...overrides
});

// node has no FileReader
class NodeFileReader {
	error: Error | null = null;
	onerror: (() => void) | null = null;
	onload: (() => void) | null = null;
	result: string | null = null;

	readAsDataURL(blob: Blob) {
		blob.arrayBuffer().then((buffer) => {
			this.result = `data:${blob.type};base64,${Buffer.from(buffer).toString('base64')}`;
			this.onload?.();
		});
	}
}

describe('findMcpToolResultVideoUrls', () => {
	it('finds the video url in an sd-server generate_video result', () => {
		expect(findMcpToolResultVideoUrls(SD_SERVER_RESULT, SERVER_URL)).toEqual([VIDEO_URL]);
	});

	it('ignores videos on other origins and links that are not videos', () => {
		const result = [
			'http://example.com/clip.webm',
			'http://192.168.1.20:9090/clip.webm',
			'https://192.168.1.20:8080/clip.webm',
			'http://192.168.1.20:8080/mcp/images/img-1a2b-2.png',
			'http://192.168.1.20:8080/clip.webm.txt'
		].join('\n');

		expect(findMcpToolResultVideoUrls(result, SERVER_URL)).toEqual([]);
	});

	it('drops trailing punctuation and brackets, keeps queries, and lists each url once', () => {
		const result = `Watch ${VIDEO_URL}. Or [here](${VIDEO_URL}), or http://192.168.1.20:8080/a.MP4?t=2!`;

		expect(findMcpToolResultVideoUrls(result, SERVER_URL)).toEqual([
			VIDEO_URL,
			'http://192.168.1.20:8080/a.MP4?t=2'
		]);
	});

	it('keeps at most the first few videos', () => {
		const result = Array.from(
			{ length: MCP_TOOL_VIDEO_MAX_COUNT + 2 },
			(_, i) => `http://192.168.1.20:8080/v${i}.webm`
		).join(' ');

		expect(findMcpToolResultVideoUrls(result, SERVER_URL)).toHaveLength(MCP_TOOL_VIDEO_MAX_COUNT);
	});

	it('returns nothing for a server url it cannot parse', () => {
		expect(findMcpToolResultVideoUrls(SD_SERVER_RESULT, 'not a url')).toEqual([]);
	});
});

describe('getMcpToolVideoMimeType', () => {
	it('maps video file extensions', () => {
		expect(getMcpToolVideoMimeType(new URL('http://h/a.webm'))).toBe('video/webm');
		expect(getMcpToolVideoMimeType(new URL('http://h/a.M4V?x=1'))).toBe('video/mp4');
		expect(getMcpToolVideoMimeType(new URL('http://h/a.ogv'))).toBe('video/ogg');
		expect(getMcpToolVideoMimeType(new URL('http://h/a.avi'))).toBeUndefined();
		expect(getMcpToolVideoMimeType(new URL('http://h/a.b/c'))).toBeUndefined();
	});
});

describe('MCPService.fetchToolVideo', () => {
	beforeEach(() => {
		vi.stubGlobal('FileReader', NodeFileReader);
	});

	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('downloads the video with the server headers and keeps it as a video attachment', async () => {
		const fetchMock = vi.fn().mockResolvedValue(
			new Response(new Uint8Array([1, 2, 3]), {
				headers: { 'content-type': 'video/webm; codecs=vp8' }
			})
		);

		vi.stubGlobal('fetch', fetchMock);

		const video = await MCPService.fetchToolVideo(
			config({ headers: { 'x-api-key': 'secret' } }),
			VIDEO_URL
		);

		expect(String(fetchMock.mock.calls[0][0])).toBe(VIDEO_URL);
		expect(fetchMock.mock.calls[0][1].headers).toEqual({ 'x-api-key': 'secret' });
		expect(video).toEqual({
			base64Data: 'AQID',
			mimeType: 'video/webm',
			name: 'img-1a2b-3.webm',
			size: 3,
			type: AttachmentType.VIDEO
		});
	});

	it('goes through the CORS proxy when the server does', async () => {
		const fetchMock = vi.fn().mockResolvedValue(new Response(new Uint8Array([1])));

		vi.stubGlobal('fetch', fetchMock);
		vi.stubGlobal('window', { location: { origin: 'http://localhost:8080' } });

		const video = await MCPService.fetchToolVideo(
			config({ headers: { 'x-api-key': 'secret' }, useProxy: true }),
			VIDEO_URL
		);
		const requested = new URL(String(fetchMock.mock.calls[0][0]));

		expect(requested.origin).toBe('http://localhost:8080');
		expect(requested.pathname).toContain(CORS_PROXY_ENDPOINT);
		expect(requested.searchParams.get(CORS_PROXY.URL_PARAM)).toBe(VIDEO_URL);
		expect(fetchMock.mock.calls[0][1].headers).toMatchObject({
			[`${CORS_PROXY.HEADER_PREFIX}x-api-key`]: 'secret'
		});
		// no video content type: the file extension decides
		expect(video.mimeType).toBe('video/webm');
	});

	it('fails on an HTTP error', async () => {
		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('gone', { status: 404 })));

		await expect(MCPService.fetchToolVideo(config(), VIDEO_URL)).rejects.toThrow('HTTP 404');
	});

	it('stops reading a file that is too large', async () => {
		const chunk = new Uint8Array(16 * 1024 * 1024);

		let sent = 0;

		const body = new ReadableStream<Uint8Array>({
			pull(controller) {
				sent += 1;
				controller.enqueue(chunk);
			}
		});

		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body)));

		await expect(MCPService.fetchToolVideo(config(), VIDEO_URL)).rejects.toThrow('larger than');
		expect(sent).toBeLessThan(8);
	});
});

describe('MCPService tool result formatting', () => {
	const formatToolResult = (result: unknown) =>
		(MCPService as unknown as { formatToolResult: (r: unknown) => string }).formatToolResult(
			result
		);

	it('turns an embedded video into a data URI line instead of bare base64', () => {
		expect(
			formatToolResult({
				content: [
					{ text: 'done', type: 'text' },
					{
						resource: { blob: 'AAAA', mimeType: 'video/webm', uri: 'sdcpp://images/img-1' },
						type: 'resource'
					}
				]
			})
		).toBe('done\ndata:video/webm;base64,AAAA');
	});
});
