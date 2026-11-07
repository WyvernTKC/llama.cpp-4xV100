/**
 * MemoryService - IndexedDB persistence for the memories the model saves
 *
 * The save_memory, search_memories and delete_memory chat tools and the
 * memory list in Settings read and write here. Memories live in their own
 * database, so the conversations schema and its migrations never change for
 * them. A new chat freezes the list into its system prompt (promptBlock).
 */

import { CHAT_TOOLS, MEMORY_IDXDB } from '$lib/constants';
import type { DatabaseMemory } from '$lib/types/database';
import { formatMemoryPromptBlock, normalizeMemoryContent, rankMemories, uuid } from '$lib/utils';
import Dexie, { type EntityTable } from 'dexie';

class MemoryDatabase extends Dexie {
	[MEMORY_IDXDB.TABLE]!: EntityTable<DatabaseMemory, 'id'>;

	constructor() {
		super(MEMORY_IDXDB.NAME);

		this.version(1).stores({ [MEMORY_IDXDB.TABLE]: MEMORY_IDXDB.SCHEMA });
	}
}

let database: MemoryDatabase | null = null;

// opened on first use, so importing the service never touches IndexedDB
function memories(): EntityTable<DatabaseMemory, 'id'> {
	database ??= new MemoryDatabase();

	return database[MEMORY_IDXDB.TABLE];
}

export class MemoryService {
	/** Deletes every memory. */
	static async clear(): Promise<void> {
		await memories().clear();
	}

	/** Every memory, newest first. */
	static async list(): Promise<DatabaseMemory[]> {
		return await memories().orderBy('updatedAt').reverse().toArray();
	}

	/** The memory list for a new chat's system prompt, empty without memories. */
	static async promptBlock(): Promise<string> {
		return formatMemoryPromptBlock(await this.list());
	}

	/**
	 * Deletes a memory by id, or by the start of one when that start is unique.
	 * @returns The deleted memory, or null when no single memory matched
	 */
	static async remove(id: string): Promise<DatabaseMemory | null> {
		const wanted = id.trim().replace(/^\[|\]$/g, '');

		if (!wanted) return null;

		const exact = await memories().get(wanted);
		const matches = exact
			? [exact]
			: await memories()
					.filter((m) => m.id.startsWith(wanted))
					.toArray();

		if (matches.length !== 1) return null;

		await memories().delete(matches[0].id);

		return matches[0];
	}

	/**
	 * Saves a memory, or returns the one already holding the same text.
	 * @throws Error when the text is empty or longer than the limit
	 */
	static async save(
		raw: unknown,
		sourceConvId?: string
	): Promise<{ created: boolean; memory: DatabaseMemory }> {
		const content = normalizeMemoryContent(raw);

		if (!content) throw new Error('The memory is empty.');

		if (content.length > CHAT_TOOLS.MEMORY_MAX_LENGTH) {
			throw new Error(
				`The memory is ${content.length} characters long; keep it under ${CHAT_TOOLS.MEMORY_MAX_LENGTH}.`
			);
		}

		const lower = content.toLowerCase();
		const existing = await memories()
			.filter((m) => m.content.toLowerCase() === lower)
			.first();

		if (existing) return { created: false, memory: existing };

		let id = '';

		do {
			id = uuid().replace(/-/g, '').slice(0, 8);
		} while (await memories().get(id));

		const now = Date.now();
		const memory: DatabaseMemory = { content, createdAt: now, id, sourceConvId, updatedAt: now };

		await memories().add(memory);

		return { created: true, memory };
	}

	/** Memories for a query, best first; an empty query lists the newest. */
	static async search(query: unknown, limit: number): Promise<DatabaseMemory[]> {
		return rankMemories(await this.list(), query, limit);
	}
}
