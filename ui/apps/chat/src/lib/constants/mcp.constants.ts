import { Globe, Radio, Zap } from '@lucide/svelte';
import { MimeTypeImage, MimeTypeVideo } from '#lib/enums/files.enums.ts';
import { MCPTransportType } from '#lib/enums/index.ts';
import type { ClientCapabilities, Implementation } from '#lib/types/index.ts';
import type { Component } from 'svelte';

export const DEFAULT_CLIENT_VERSION = '1.0.0';
export const MCP_CLIENT_NAME = 'llama-ui-mcp';
export const DEFAULT_IMAGE_MIME_TYPE = MimeTypeImage.PNG;

/** MIME types considered safe for rendering MCP server icons */
export const MCP_ALLOWED_ICON_MIME_TYPES = new Set([
	MimeTypeImage.PNG,
	MimeTypeImage.JPEG,
	MimeTypeImage.JPG,
	MimeTypeImage.SVG,
	MimeTypeImage.WEBP,
	MimeTypeImage.ICO,
	MimeTypeImage.ICO_MICROSOFT
]);

/**
 * MCP specification version this client targets.
 * Update when the upstream MCP spec introduces a new stable version:
 * https://spec.modelcontextprotocol.io/
 */
export const MCP_PROTOCOL_VERSION = '2025-06-18';

export const DEFAULT_MCP_CONFIG = {
	capabilities: { tools: { listChanged: true } } as ClientCapabilities,
	clientInfo: { name: MCP_CLIENT_NAME, version: DEFAULT_CLIENT_VERSION } as Implementation,
	connectionTimeoutMs: 10_000, // 10 seconds for connection establishment
	protocolVersion: MCP_PROTOCOL_VERSION,
	requestTimeoutSeconds: 300 // 5 minutes for long-running tools
} as const;

export const MCP_SERVER_ID_PREFIX = 'LlamaUI-MCP-Server';

/**
 * Progress notification `_meta` key under which the stable-diffusion.cpp server
 * sends its job state and a live preview of the image being made.
 */
export const MCP_PROGRESS_META_KEY = 'sdcpp/progress';

/** Largest progress preview the UI shows, in base64 characters */
export const MCP_PROGRESS_PREVIEW_MAX_BASE64_CHARS = 1_000_000;

/** Raster image types a progress preview may use */
export const MCP_PROGRESS_PREVIEW_MIME_TYPES = new Set<string>([
	MimeTypeImage.PNG,
	MimeTypeImage.JPEG,
	MimeTypeImage.WEBP
]);

/** Video files a tool result can link (by file extension) for the UI to download and play */
export const MCP_TOOL_VIDEO_EXTENSION_TO_MIME: Record<string, string> = {
	m4v: MimeTypeVideo.MP4,
	mp4: MimeTypeVideo.MP4,
	ogv: MimeTypeVideo.OGG,
	webm: MimeTypeVideo.WEBM
};

/** Most linked videos the UI downloads from one tool result */
export const MCP_TOOL_VIDEO_MAX_COUNT = 4;

/** Largest linked video the UI downloads, in bytes */
export const MCP_TOOL_VIDEO_MAX_BYTES = 64 * 1024 * 1024;

/** http(s) URLs in tool result text: ends at whitespace, quotes or brackets, never on punctuation */
export const MCP_TOOL_RESULT_URL_REGEX =
	/https?:\/\/[^\s"'`<>()[\]{}\\]*[^\s"'`<>()[\]{}\\.,;:!?]/gi;

/** Backoff policy for reconnecting to a dropped MCP server. */
export const MCP_RECONNECT = {
	/** Per-attempt timeout for a single reconnection attempt before giving up and backing off. */
	ATTEMPT_TIMEOUT_MS: 15_000,
	BACKOFF_MULTIPLIER: 2,
	INITIAL_DELAY: 1000,
	MAX_DELAY: 30000
};

/** Maximum number of MCP server avatars to display in the chat form */
export const MAX_DISPLAYED_MCP_AVATARS = 4;

/** Expected count when two theme-less icons represent a light/dark pair */
export const EXPECTED_THEMED_ICON_PAIR_COUNT = 2;

/** CORS proxy connection settings */
export const CORS_PROXY = {
	/** Header prefix for headers that should be forwarded by the CORS proxy */
	HEADER_PREFIX: 'x-llama-server-proxy-header-',
	/** CORS proxy URL query parameter name */
	URL_PARAM: 'url'
} as const;

/** Standard SSE endpoint path indicators */
export const MCP_SSE = {
	ENDPOINT: '/sse',
	ENDPOINT_QUERY: '/sse?',
	ENDPOINT_SLASH: '/sse/'
} as const;

/** Human-readable labels for MCP transport types */
export const MCP_TRANSPORT_LABELS: Record<MCPTransportType, string> = {
	[MCPTransportType.SSE]: 'SSE',
	[MCPTransportType.STREAMABLE_HTTP]: 'HTTP',
	[MCPTransportType.WEBSOCKET]: 'WebSocket'
};

/** Icon components for MCP transport types */
export const MCP_TRANSPORT_ICONS: Record<MCPTransportType, Component> = {
	[MCPTransportType.SSE]: Radio,
	[MCPTransportType.STREAMABLE_HTTP]: Globe,
	[MCPTransportType.WEBSOCKET]: Zap
};
