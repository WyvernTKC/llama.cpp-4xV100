import {
	ATTACHMENT_REF_PREFIX,
	ATTACHMENT_REF_REGEX,
	BASE64_HINT_REGEX,
	DATA_URI_HINT_REGEX
} from '$lib/constants';
import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types/database';

type JsonSchema = Record<string, unknown>;

/** Text part sent to the model next to a chat image, so it can name the image in a tool call. */
export function formatAttachmentRefLabel(refId: string): string {
	return `[Image ${ATTACHMENT_REF_PREFIX}${refId} - pass this handle in a tool argument to send the image]`;
}

/** Data URL of every chat image that carries a ref id, keyed by that id. */
export function collectAttachmentRefs(messages: readonly object[]): Map<string, string> {
	const refs = new Map<string, string>();

	for (const message of messages) {
		const extras = 'extra' in message ? (message.extra as DatabaseMessageExtra[] | undefined) : [];

		for (const extra of extras ?? []) {
			if (extra.type === AttachmentType.IMAGE && extra.refId) {
				refs.set(extra.refId, extra.base64Url);
			}
		}
	}

	return refs;
}

function asSchema(value: unknown): JsonSchema | undefined {
	return value && typeof value === 'object' ? (value as JsonSchema) : undefined;
}

function wantsRawBase64(schema: JsonSchema | undefined, key: string | undefined): boolean {
	if (schema?.contentEncoding === 'base64') return true;

	const hint = [key, schema?.title, schema?.description]
		.filter((part) => typeof part === 'string')
		.join(' ');

	if (DATA_URI_HINT_REGEX.test(hint)) return false;

	return BASE64_HINT_REGEX.test(hint);
}

function resolveValue(
	value: unknown,
	schema: JsonSchema | undefined,
	key: string | undefined,
	refs: Map<string, string>
): unknown {
	if (typeof value === 'string') {
		const match = value.trim().match(ATTACHMENT_REF_REGEX);

		if (!match) return value;

		const dataUrl = refs.get(match[1]);

		if (!dataUrl) {
			throw new Error(`Unknown image handle "${value}", use one from an image label in the chat`);
		}

		return wantsRawBase64(schema, key) ? dataUrl.slice(dataUrl.indexOf(',') + 1) : dataUrl;
	}

	if (Array.isArray(value)) {
		return value.map((item) => resolveValue(item, asSchema(schema?.items), key, refs));
	}

	if (value && typeof value === 'object') {
		const properties = asSchema(schema?.properties);

		return Object.fromEntries(
			Object.entries(value).map(([k, v]) => [
				k,
				resolveValue(v, asSchema(properties?.[k]), k, refs)
			])
		);
	}

	return value;
}

/**
 * Replace attachment://<refId> strings in tool arguments with the image they name:
 * raw base64 when the parameter schema asks for base64, a data URI otherwise.
 * Arguments without a handle come back untouched.
 */
export function resolveAttachmentRefs(
	args: string | Record<string, unknown>,
	parameters: JsonSchema | undefined,
	refs: Map<string, string>
): string | Record<string, unknown> {
	if (typeof args === 'string' && !args.includes(ATTACHMENT_REF_PREFIX)) return args;

	const parsed = typeof args === 'string' ? JSON.parse(args) : args;

	return resolveValue(parsed, parameters, undefined, refs) as Record<string, unknown>;
}
