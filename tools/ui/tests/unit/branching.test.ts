// Sibling-info correctness for buildSiblingInfoMap, including the memoized
// leaf resolution. A wrong leaf id here breaks branch navigation, so the
// deep-chain and multi-branch cases below pin the resolution down.

import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types/database';
import { buildSiblingInfoMap, filterByLeafNodeId, findLeafNode } from '$lib/utils/branching';
import { describe, expect, it } from 'vitest';

function msg(id: string, parent: string | null, children: string[] = []): DatabaseMessage {
	return {
		children,
		content: '',
		convId: 'c1',
		id,
		parent,
		role: MessageRole.USER,
		timestamp: 0,
		type: MessageType.TEXT
	} as DatabaseMessage;
}

/** root -> m1 -> ... -> m depth, each node with a single child. */
function linearChain(depth: number): DatabaseMessage[] {
	const messages = [msg('m0', null, ['m1'])];

	for (let i = 1; i <= depth; i++) {
		messages.push(msg(`m${i}`, `m${i - 1}`, i < depth ? [`m${i + 1}`] : []));
	}

	return messages;
}

describe('buildSiblingInfoMap', () => {
	it('resolves the deepest leaf for every node of a long single chain', () => {
		const messages = linearChain(50);
		const map = buildSiblingInfoMap(messages);
		const leafId = messages[messages.length - 1].id;

		// every non-root message of the chain is an only child, and its
		// navigation target is the chain's deepest leaf
		for (const m of messages.slice(1)) {
			const info = map.get(m.id);

			expect(info?.totalSiblings).toBe(1);
			expect(info?.siblingIds).toEqual([leafId]);
		}
	});

	it('reports sibling position and leaf targets on a branched tree', () => {
		// m0 -> m1, m4 ; m1 -> m2 ; m2 -> m3, m6 ; m4 -> m5
		const root = msg('m0', null, ['m1', 'm4']);
		const m1 = msg('m1', 'm0', ['m2']);
		const m2 = msg('m2', 'm1', ['m3', 'm6']);
		const m3 = msg('m3', 'm2');
		const m4 = msg('m4', 'm0', ['m5']);
		const m5 = msg('m5', 'm4');
		const m6 = msg('m6', 'm2');
		const map = buildSiblingInfoMap([root, m1, m2, m3, m4, m5, m6]);

		// m1 and m4 share the root as parent; their nav targets are the
		// leaves of their subtrees ( m6 for the first branch, m5 for the second )
		expect(map.get(m1.id)).toMatchObject({
			currentIndex: 0,
			siblingIds: [m6.id, m5.id],
			totalSiblings: 2
		});
		expect(map.get(m4.id)).toMatchObject({
			currentIndex: 1,
			siblingIds: [m6.id, m5.id],
			totalSiblings: 2
		});

		// m3 and m6 are siblings under m2; both are leaves
		expect(map.get(m3.id)?.siblingIds).toEqual([m3.id, m6.id]);
		expect(map.get(m6.id)?.currentIndex).toBe(1);

		// the root has no parent and reports itself
		expect(map.get(root.id)).toMatchObject({
			currentIndex: 0,
			siblingIds: [root.id],
			totalSiblings: 1
		});
	});

	it('agrees with findLeafNode for arbitrary nodes', () => {
		const messages = linearChain(20);
		const leafId = messages[messages.length - 1].id;

		// every node of the chain resolves to the deepest leaf
		for (const m of messages) {
			expect(findLeafNode(messages, m.id), `leaf of ${m.id}`).toBe(leafId);
		}
	});
});

/** A message with a role, an optional tool call id and a timestamp for path order. */
function node(
	id: string,
	parent: string | null,
	children: string[],
	role: MessageRole,
	timestamp: number,
	toolCallId?: string
): DatabaseMessage {
	return { ...msg(id, parent, children), role, timestamp, toolCallId } as DatabaseMessage;
}

describe('filterByLeafNodeId with parallel tool calls', () => {
	it('keeps every result of a chained turn', () => {
		// u1 -> a1 (two calls) -> t1 -> t2 -> a2, as new flows store them
		const messages = [
			node('u1', null, ['a1'], MessageRole.USER, 1),
			node('a1', 'u1', ['t1'], MessageRole.ASSISTANT, 2),
			node('t1', 'a1', ['t2'], MessageRole.TOOL, 3, 'call_1'),
			node('t2', 't1', ['a2'], MessageRole.TOOL, 4, 'call_2'),
			node('a2', 't2', [], MessageRole.ASSISTANT, 5)
		];

		expect(filterByLeafNodeId(messages, 'a2').map((m) => m.id)).toEqual([
			'u1',
			'a1',
			't1',
			't2',
			'a2'
		]);
	});

	it('recovers results stored as siblings under the assistant message', () => {
		// older chats: both results hang off a1, the next turn off the last one
		const messages = [
			node('u1', null, ['a1'], MessageRole.USER, 1),
			node('a1', 'u1', ['t1', 't2'], MessageRole.ASSISTANT, 2),
			node('t1', 'a1', [], MessageRole.TOOL, 3, 'call_1'),
			node('t2', 'a1', ['a2'], MessageRole.TOOL, 4, 'call_2'),
			node('a2', 't2', [], MessageRole.ASSISTANT, 5)
		];

		expect(filterByLeafNodeId(messages, 'a2').map((m) => m.id)).toEqual([
			'u1',
			'a1',
			't1',
			't2',
			'a2'
		]);
	});

	it('leaves out user branches, branched tool results and duplicate call ids', () => {
		const messages = [
			node('u1', null, ['a1'], MessageRole.USER, 1),
			node('a1', 'u1', ['t1', 'tdup', 'tb', 'u2', 't2'], MessageRole.ASSISTANT, 2),
			node('t1', 'a1', [], MessageRole.TOOL, 3, 'call_1'),
			node('tdup', 'a1', [], MessageRole.TOOL, 4, 'call_2'),
			node('tb', 'a1', ['x'], MessageRole.TOOL, 5, 'call_3'),
			node('x', 'tb', [], MessageRole.ASSISTANT, 6),
			node('u2', 'a1', [], MessageRole.USER, 7),
			node('t2', 'a1', ['a2'], MessageRole.TOOL, 8, 'call_2'),
			node('a2', 't2', [], MessageRole.ASSISTANT, 9)
		];

		expect(filterByLeafNodeId(messages, 'a2').map((m) => m.id)).toEqual([
			'u1',
			'a1',
			't1',
			't2',
			'a2'
		]);
	});
});
