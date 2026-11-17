import { uuid } from './uuid';
import {
	ATTACHMENT_REF_PARAM_HINT,
	ATTACHMENT_REF_PREFIX,
	ATTACHMENT_REF_REGEX,
	ATTACHMENT_SAVED_REGEX,
	BASE64_HINT_REGEX,
	DATA_URI_HINT_REGEX,
	IMAGE_PARAM_HINT_REGEX,
	NEWLINE
} from '$lib/constants';
import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtra, DatabaseMessageExtraImageFile } from '$lib/types/database';

type JsonSchema = Record<string, unknown>;

/** Text part sent to the model next to a chat image, so it can name the image in a tool call. */
export function formatAttachmentRefLabel(refId: string): string {
	return `[Image ${ATTACHMENT_REF_PREFIX}${refId} - pass this handle in a tool argument to send the image]`;
}

/**
 * Label of an image a tool returned. Templates that render tool results in a
 * user turn (Qwen) make an image there look like a new upload when it carries
 * the upload label, and the model reruns the user's last request on it.
 */
export function formatToolImageRefLabel(refId: string): string {
	return `[Image ${ATTACHMENT_REF_PREFIX}${refId} - returned by this tool call, not sent by the user; pass this handle in a tool argument to send the image]`;
}

/** Data URL of every chat image that carries a ref id, PDF pages sent as images too, keyed by that id. */
export function collectAttachmentRefs(messages: readonly object[]): Map<string, string> {
	const refs = new Map<string, string>();

	for (const message of messages) {
		const extras = 'extra' in message ? (message.extra as DatabaseMessageExtra[] | undefined) : [];

		for (const extra of extras ?? []) {
			if (extra.type === AttachmentType.IMAGE && extra.refId) {
				refs.set(extra.refId, extra.base64Url);
			} else if (extra.type === AttachmentType.PDF && extra.processedAsImages) {
				extra.images?.forEach((image, i) => {
					const refId = extra.imageRefIds?.[i];

					if (refId) refs.set(refId, image);
				});
			}
		}
	}

	return refs;
}

/**
 * Gives every image on the path a ref id no earlier image on it holds: images
 * saved before ref ids existed get one, and so does a copy of an earlier image.
 * PDF pages sent as images count as images. Returns the new extras of each
 * message that changed, keyed by message id; the other messages keep theirs.
 */
export function assignImageRefIds(
	messages: readonly { id: string; extra?: DatabaseMessageExtra[] }[],
	newId: () => string = uuid
): Map<string, DatabaseMessageExtra[]> {
	const seen = new Set<string>();
	const changed = new Map<string, DatabaseMessageExtra[]>();
	const claim = (refId: string | undefined): string => {
		const id = refId && !seen.has(refId) ? refId : newId();

		seen.add(id);

		return id;
	};

	for (const message of messages) {
		if (!message.extra?.length) continue;

		let dirty = false;

		const extra = message.extra.map((item): DatabaseMessageExtra => {
			if (item.type === AttachmentType.IMAGE) {
				const refId = claim(item.refId);

				if (refId === item.refId) return item;

				dirty = true;

				return { ...item, refId };
			}

			if (item.type === AttachmentType.PDF && item.processedAsImages && item.images?.length) {
				const imageRefIds = item.images.map((_, i) => claim(item.imageRefIds?.[i]));

				if (imageRefIds.every((id, i) => id === item.imageRefIds?.[i])) return item;

				dirty = true;

				return { ...item, imageRefIds };
			}

			return item;
		});

		if (dirty) changed.set(message.id, extra);
	}

	return changed;
}

/** The extras with a new ref id on each image, for a copy of them in a new message. */
export function withFreshImageRefIds<T extends DatabaseMessageExtra[] | undefined>(
	extras: T,
	newId: () => string = uuid
): T {
	if (!extras) return extras;

	return extras.map((item) => {
		if (item.type === AttachmentType.IMAGE) return { ...item, refId: newId() };

		if (item.type === AttachmentType.PDF && item.processedAsImages && item.images?.length) {
			return { ...item, imageRefIds: item.images.map(() => newId()) };
		}

		return item;
	}) as T;
}

/**
 * The text of a tool result as the model reads it: each image the tool returned
 * keeps the "[Attachment saved: name]" line it was swapped for, and gets its
 * attachment://<refId> label on that line, worded as tool output. An image without such a line gets
 * its label at the end. The stored text stays as is, the chat view parses it.
 */
export function withToolImageHandles(
	content: string,
	extras: readonly DatabaseMessageExtra[] | undefined
): string {
	const images = (extras ?? []).filter(
		(e): e is DatabaseMessageExtraImageFile & { refId: string } =>
			e.type === AttachmentType.IMAGE && Boolean(e.refId)
	);

	if (images.length === 0) return content;

	// older chats stored each tool image twice under one name
	const byName = new Map<string, string>();

	for (const image of images) {
		if (!byName.has(image.name)) byName.set(image.name, image.refId);
	}

	const labelled = new Set<string>();
	const lines = content.split(NEWLINE).map((line) => {
		const name = line.match(ATTACHMENT_SAVED_REGEX)?.[1];
		const refId = name === undefined ? undefined : byName.get(name);

		if (!name || !refId || labelled.has(name)) return line;

		labelled.add(name);

		return `${line} ${formatToolImageRefLabel(refId)}`;
	});

	for (const [name, refId] of byName) {
		if (!labelled.has(name)) lines.push(formatToolImageRefLabel(refId));
	}

	return lines.join(NEWLINE);
}

function asSchema(value: unknown): JsonSchema | undefined {
	return value && typeof value === 'object' ? (value as JsonSchema) : undefined;
}

function hintText(schema: JsonSchema | undefined, key: string | undefined): string {
	return [key, schema?.title, schema?.description]
		.filter((part) => typeof part === 'string')
		.join(' ');
}

function wantsRawBase64(schema: JsonSchema | undefined, key: string | undefined): boolean {
	if (schema?.contentEncoding === 'base64') return true;

	const hint = hintText(schema, key);

	if (DATA_URI_HINT_REGEX.test(hint)) return false;

	return BASE64_HINT_REGEX.test(hint);
}

function hasType(schema: JsonSchema | undefined, type: string): boolean {
	const types = schema?.type;

	return Array.isArray(types) ? types.includes(type) : types === type;
}

function hintProperty(key: string, property: JsonSchema): JsonSchema {
	if (hasType(property, 'object')) return withAttachmentRefHints(property);

	const takesStrings =
		hasType(property, 'string') ||
		(hasType(property, 'array') && hasType(asSchema(property.items), 'string'));
	const name = [key, property.title].filter((part) => typeof part === 'string').join(' ');

	if (!takesStrings || property.enum || !IMAGE_PARAM_HINT_REGEX.test(name)) return property;

	const description = typeof property.description === 'string' ? `${property.description} ` : '';

	return { ...property, description: description + ATTACHMENT_REF_PARAM_HINT };
}

/**
 * Tell the model, on each string parameter that takes an image, that a chat image's
 * attachment://<refId> handle works there. Depends on the schema only, so the tool
 * list stays byte-identical across turns.
 */
export function withAttachmentRefHints(schema: JsonSchema): JsonSchema {
	const properties = asSchema(schema.properties);

	if (!properties) return schema;

	return {
		...schema,
		properties: Object.fromEntries(
			Object.entries(properties).map(([key, value]) => {
				const property = asSchema(value);

				return [key, property ? hintProperty(key, property) : value];
			})
		)
	};
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
