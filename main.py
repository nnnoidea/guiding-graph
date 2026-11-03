"""Freebase question answering guided by a question-derived graph.

Retrieval combines both edge directions. Structural pruning falls back to
semantic relation selection when candidate neighborhoods are too large.
"""

import copy
import json
import logging
import re
import time
from pathlib import Path

from utils import (
    answer_question, chat, choose, complete_info, configure, count_nodes,
    get_related_nodes, get_related_nodes_by_id, get_relations,
    prepare_dataset, prune_node, query_node_by_name, structure,
)

# Edit these settings before running. Paths are relative to this file.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WebQSP.json"
OUTPUT_PATH = ROOT / "results" / "webqsp.jsonl"
SAMPLE_LIMIT = None  # None: process the entire dataset.
MAX_ANSWER_TRIPLES = 30  # None: provide all retrieved triples.
ALLOW_OWN_KNOWLEDGE = True  # Allow model knowledge when retrieved evidence is insufficient.

logger = logging.getLogger(__name__)

class Node:
    """A KG candidate; relations maps neighbor IDs to predicates."""

    def __init__(self, name, id, clue, label=None, relations: dict[str, str]=None):
        self.name = name
        self.id = id
        self.clue = clue
        self.label = label
        if relations is not None:
            self.relations = relations
        else:
            self.relations = {}
        self.related_id = []
        self.relation = []
        if relations is not None:
            for related_id, relation in relations.items():
                self.related_id.append(related_id)
                self.relation.append(relation)

class NodeChain:
    """Candidate batches organized by question clues and their dependencies."""

    def __init__(self, init_node, pairs, set_clue=None):
        self.batches = [[init_node]]
        self.clues = [init_node.clue]
        self.pairs = pairs
        self.checked_batches = set()
        self.explored_clues = {init_node.clue}
        self.next_clue, self.current_clue, self.current_pair = self._get_next_clue(set_clue=set_clue)
        self.potential_clues = self._get_potential_clues()

    def add_batch(self, new_nodes):
        # A failed target is marked visited so exploration can try another clue.
        if not new_nodes:
            self.explored_clues.add(self.next_clue)
            self.next_clue, self.current_clue, self.current_pair = self._get_next_clue()
            return
        new_clue = new_nodes[0].clue
        if new_clue in self.clues:
            raise ValueError(f"Clue '{new_clue}' already exists in a previous batch.")
        self.explored_clues.add(self.next_clue)
        self.batches.append(new_nodes)
        self.clues.append(new_clue)
        self.checked_batches = set()
        self._prune_unrelated_nodes()
        self.next_clue, self.current_clue, self.current_pair = self._get_next_clue()

    def generate_triples(self):
        # Assemble evidence from candidate links between question-graph batches.
        triples = set()
        for clue1, clue2 in self.pairs:
            batch1 = self._find_batch_by_clue(clue1)
            if batch1 is None:
                continue
            batch2 = self._find_batch_by_clue(clue2)
            if batch2 is None:
                continue
            for node1 in batch1:
                for node2 in batch2:
                    if node1.id in node2.relations:
                        relation = node2.relations[node1.id]
                        triples.add((node1.name, relation, node2.name))
                    elif node2.id in node1.relations:
                        relation = node1.relations[node2.id]
                        triples.add((node2.name, relation, node1.name))
        return list(triples)

    def _get_next_clue(self, set_clue=None):
        # Follow a question-graph edge from a grounded clue to an unvisited one.
        for pair in self.pairs:
            if set_clue is not None:
                if set_clue not in pair:
                    continue
            if pair[0] in self.clues and pair[1] not in self.clues and (pair[1] not in self.explored_clues):
                return (pair[1], pair[0], pair)
            elif pair[1] in self.clues and pair[0] not in self.clues and (pair[0] not in self.explored_clues):
                return (pair[0], pair[1], pair)
        return (None, None, None)

    def _get_potential_clues(self):
        potential_clues = set()
        if self.next_clue is not None:
            all_clues = set()
            for pair in self.pairs:
                all_clues.add(pair[0])
                all_clues.add(pair[1])
            potential_clues = all_clues - set(self.clues) - set([self.next_clue])
        return list(potential_clues)

    def check_multi_branch(self):
        # Collect grounded neighbors that constrain the same next target.
        clues = set()
        clue2nodes = {}
        for pair in self.pairs:
            if self.next_clue == pair[0] and pair[1] in self.clues:
                clues.add(pair[1])
            elif self.next_clue == pair[1] and pair[0] in self.clues:
                clues.add(pair[0])
        if len(clues) > 1:
            for clue in clues:
                clue2nodes[clue] = copy.deepcopy(self._find_batch_by_clue(clue))
            return clue2nodes
        return None

    def _prune_unrelated_nodes(self):
        if not self.pairs:
            return
        for i in range(len(self.batches) - 1, -1, -1):
            if i in self.checked_batches:
                continue
            batch = self.batches[i]
            clue = self.clues[i]
            related_batches = []
            for pair in self.pairs:
                if clue == pair[0]:
                    related_batches.append(self._find_batch_by_clue(pair[1]))
                elif clue == pair[1]:
                    related_batches.append(self._find_batch_by_clue(pair[0]))
            if not related_batches:
                continue
            for related_batch in related_batches:
                if related_batch is not None:
                    self._check_and_prune_nodes(batch, related_batch)
            self.checked_batches.add(i)
        # Defer deletion so pruning does not invalidate the traversal indices.
        nonempty = [(clue, batch) for clue, batch in zip(self.clues, self.batches) if batch]
        self.clues = [clue for clue, _ in nonempty]
        self.batches = [batch for _, batch in nonempty]
        self.checked_batches.clear()

    def _find_batch_by_clue(self, clue):
        for i, c in enumerate(self.clues):
            if c == clue:
                return self.batches[i]
        return None

    def _check_and_prune_nodes(self, batch1, batch2):
        for node2 in batch2[:]:
            has_relation = False
            for node1 in batch1:
                if node2.id in node1.related_id or node1.id in node2.related_id:
                    has_relation = True
                    break
            if not has_relation:
                batch2.remove(node2)
    def __repr__(self):
        return f'NodeChain(batches={self.batches}, clues={self.clues}, pairs={self.pairs})'

class Exploration:
    """Question-guided exploration with structural pruning and semantic fallbacks."""

    def __init__(self, init_nodes, pairs, triples, target, generic, statement):
        self.init_clue2node = {node.clue: node for node in init_nodes}
        self.chain = NodeChain(init_nodes[0], pairs)
        self.explored_init = [init_nodes[0].clue]
        self.pairs = pairs
        self.triples = triples
        self.target = target
        self.generic = generic
        self.statement = statement
        self.llm_call = 0
        self.next_flag = 0
        self.exceed = 0

    def explore(self):
        input_token = 0
        output_token = 0
        total_token = 0
        next_clue = self.chain.next_clue
        current_clue = self.chain.current_clue
        current_batch = self.chain._find_batch_by_clue(current_clue)
        if next_clue in self.explored_init:
            return ([], {'input': input_token, 'output': output_token, 'total': total_token})
        # A known target can be checked directly against the current candidates.
        if next_clue in self.init_clue2node.keys() and next_clue not in self.explored_init:
            new_batch, relations_return = self.intersection([current_batch], next_clue)
            if new_batch:
                return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})
            else:
                new_batch, token_num = self.llm_prune(self.chain)
                input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                new_batch, token_num = self.check_unnamed_entity(new_batch)
                input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})
        # Another anchored entity may constrain the target before it is expanded.
        for init_node in self.init_clue2node.values():
            if init_node.clue not in self.chain.clues and init_node.clue not in self.chain.explored_clues and (init_node.clue not in self.explored_init) and ((next_clue, init_node.clue) in self.pairs or (init_node.clue, next_clue) in self.pairs):
                batches = [current_batch, [init_node]]
                new_batch, relations_return = self.intersection(batches, next_clue)
                batch1 = []
                batch2 = []
                if new_batch:
                    r_current = []
                    r_init = []
                    current_ids = [node.id for node in current_batch]
                    for relation in relations_return:
                        if relation[1] == init_node.id:
                            r_init.append(relation)
                        elif relation[1] in current_ids:
                            r_current.append(relation)
                    if len(r_current) > 1:
                        batch1, token_num = self.llm_prune(new_batch, r_current, self.chain)
                        input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                        batch1, token_num = self.check_unnamed_entity(batch1)
                        input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    else:
                        batch1 = new_batch.copy()
                    if len(r_init) > 1:
                        batch2, token_num = self.llm_prune(new_batch, r_init, self.chain)
                        input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                        batch2, token_num = self.check_unnamed_entity(batch2)
                        input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    else:
                        batch2 = new_batch.copy()
                    new_batch = []
                    for node in batch1:
                        for node2 in batch2:
                            if node.id == node2.id:
                                new_batch.append(node)
                if not new_batch:
                    batch1, token_num = self.llm_prune(self.chain)
                    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    batch1, token_num = self.check_unnamed_entity(batch1)
                    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    tmp_chain = NodeChain(init_node, self.pairs, set_clue=next_clue)
                    batch2, token_num = self.llm_prune(tmp_chain)
                    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    batch2, token_num = self.check_unnamed_entity(batch2, chain=tmp_chain)
                    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    # Empty branches have no relation to compare.
                    if not batch1 or not batch2:
                        if batch2:
                            self.explored_init.append(tmp_chain.current_clue)
                            self.chain = tmp_chain
                        elif batch1:
                            self.chain.explored_clues.add(tmp_chain.current_clue)
                            self.explored_init.append(tmp_chain.current_clue)
                        return (batch1 or batch2, {'input': input_token, 'output': output_token, 'total': total_token})
                    str1 = self.chain.current_pair[0] + ' ' + batch1[0].relation[0] + ' ' + self.chain.current_pair[1]
                    str2 = tmp_chain.current_pair[0] + ' ' + batch2[0].relation[0] + ' ' + tmp_chain.current_pair[1]
                    choosen, token_num = self.choose_one(str1, str2, tmp_chain)
                    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    if choosen == 1:
                        new_batch = batch1
                    elif choosen == 2:
                        new_batch = batch2
                    else:
                        return ([], {'input': input_token, 'output': output_token, 'total': total_token})
                return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})
        # Intersect multiple grounded branches before semantic relation scoring.
        clue2nodes = self.chain.check_multi_branch()
        if clue2nodes:
            batches = list(clue2nodes.values())
            new_batch, relations_return = self.intersection(batches, next_clue)
            if new_batch:
                if len(new_batch) == 1:
                    return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})
                else:
                    new_batch, token_num = self.llm_prune(new_batch, relations_return, self.chain)
                    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    new_batch, token_num = self.check_unnamed_entity(new_batch)
                    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                    return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})
            else:
                new_batch, token_num = self.llm_prune(self.chain)
                input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                new_batch, token_num = self.check_unnamed_entity(new_batch)
                input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})
        new_batch, token_num = self.llm_prune(self.chain)
        input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
        new_batch, token_num = self.check_unnamed_entity(new_batch)
        input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
        return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})

    def choose_one(self, batch1_clue_triples, batch2_clue_triples, tmp_chain):
        choosen, token_num = choose(self.statement, batch1_clue_triples, batch2_clue_triples)
        if choosen == '2':
            self.explored_init.append(tmp_chain.current_clue)
            self.chain = tmp_chain
            return (2, token_num)
        elif choosen == '1':
            self.chain.explored_clues.add(tmp_chain.current_clue)
            self.explored_init.append(tmp_chain.current_clue)
            return (1, token_num)
        else:
            return (0, token_num)

    def check_unnamed_entity(self, batch, chain=None):
        """Expand unnamed nodes under the branch currently being evaluated."""
        chain = self.chain if chain is None else chain
        related_nodes = []
        new_batch = []
        token_num = {'input': 0, 'output': 0, 'total': 0}
        if batch == []:
            return ([], token_num)
        for node in batch:
            if node.name == 'UnName_Entity':
                tmp = get_related_nodes_by_id(node.id)
                tmp = [(t[0], t[1], t[2], node.relations) for t in tmp if t[2] not in node.related_id]
                for t in tmp:
                    if t[0] in node.relation:
                        new_batch.append(Node(name=t[1], id=t[2], clue=node.clue, relations=t[3]))
                    else:
                        related_nodes.append(t)
            else:
                new_batch.append(node)
        related_nodes = [node for node in related_nodes if node[2] not in [n.id for n in new_batch]]
        if related_nodes:
            UnName_relations = [node[0] for node in related_nodes]
            UnName_relations = list(set(UnName_relations))
            clue = batch[0].clue
            matched_clues = []
            pair = chain.current_pair
            for triple in self.triples:
                if pair[0] == triple[0] and pair[1] == triple[2]:
                    if clue == pair[0]:
                        matched_clues.append(f'the target {clue}_{triple[1]}_{triple[2]}')
                    elif clue == pair[1]:
                        matched_clues.append(f'{triple[0]}_{triple[1]}_the target {clue}')
                    break
            UnName_relations_result, token_num = prune_node(matched_clues, UnName_relations, {})
            UnName_relations_pruned = UnName_relations_result.get(matched_clues[0], [])
            self.llm_call += 1
            if UnName_relations_pruned != []:
                for UnName_relation in UnName_relations_pruned:
                    for node in related_nodes:
                        if node[0] == UnName_relation[0]:
                            new_batch.append(Node(name=node[1], id=node[2], clue=clue, relations=node[3]))
        return (new_batch, token_num)

    def llm_prune(self, *args):
        """Accept a chain for expansion, or candidates, relation_info, chain for filtering."""
        relations_info = []
        batch = []
        input_token = 0
        output_token = 0
        total_token = 0
        # One argument expands an anchor; three arguments filter known candidates.
        if len(args) == 1:
            chain = args[0]
            batch = chain._find_batch_by_clue(chain.current_clue)
            for node in batch:
                tmp = get_relations(node.id)
                relations_info.extend(tmp)
                relations_info = list(set(relations_info))
        elif len(args) == 3:
            batch, relations_info, chain = args
        else:
            raise ValueError('Invalid number of arguments. Expected 1 or 3 arguments.')
        relations = [info[0] for info in relations_info]
        if relations == []:
            return ([], {'input': input_token, 'output': output_token, 'total': total_token})
        relations = list(set(relations))
        candidate_clues = []
        if len(args) == 3:
            candidate_clues = [chain.next_clue]
        else:
            candidate_clues = [chain.next_clue]
        matched_clues = []
        # Scope relation matching to the current triple in the question graph.
        clue = chain.next_clue
        pair = chain.current_pair
        for triple in self.triples:
            if pair[0] == triple[0] and pair[1] == triple[2]:
                if clue == pair[0]:
                    matched_clues.append(f'the target {clue}_{triple[1]}_{triple[2]}')
                elif clue == pair[1]:
                    matched_clues.append(f'{triple[0]}_{triple[1]}_the target {clue}')
                break
        pruned_relations = {}
        prune_once_more = 0
        if len(relations) / 20 + self.llm_call > 30:
            self.exceed = 1
            return ([], {'input': input_token, 'output': output_token, 'total': total_token})
        while len(relations) > 0:
            prune_flag = 0
            if len(relations) > 20:
                prune_once_more = 1
                relations_for_match = relations[:20]
                relations = relations[20:]
            else:
                relations_for_match = relations
                prune_flag = 1
            pruned_relations, token_num = prune_node(matched_clues, relations_for_match, pruned_relations)
            input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
            self.llm_call += 1
            if prune_flag == 1:
                break
        if pruned_relations == {}:
            return ([], {'input': input_token, 'output': output_token, 'total': total_token})
        if prune_once_more == 1:
            relations_for_match = []
            matched_clues = []
            for clue in pruned_relations.keys():
                if 'the target ' + chain.next_clue in clue:
                    relations_for_match = pruned_relations[clue]
                    relations_for_match = [rel[0] for rel in relations_for_match]
                    matched_clues.append(clue)
                    break
            pruned_relations = {}
            if relations_for_match:
                pruned_relations, token_num = prune_node(matched_clues, relations_for_match, pruned_relations)
                input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, token_num)
                self.llm_call += 1
        top_relations = []
        top_clue = ''
        for clue_r, relations in pruned_relations.items():
            clues = clue_r.split('_')
            if len(clues) != 3:
                continue
            if 'the target' in clues[0]:
                top_clue = clues[0].replace('the target ', '')
                if top_clue in candidate_clues:
                    max_value = max(relations, key=lambda x: x[2])[2]
                    top_relations = [rel[0] for rel in relations if rel[2] == max_value]
            elif 'the target' in clues[2]:
                top_clue = clues[2].replace('the target ', '')
                if top_clue in candidate_clues:
                    max_value = max(relations, key=lambda x: x[2])[2]
                    top_relations = [rel[0] for rel in relations if rel[2] == max_value]
        new_batch = []
        if len(args) == 3:
            for node in batch:
                for top_relation in top_relations:
                    if top_relation in node.relation:
                        new_batch.append(node)
        else:
            for info in relations_info:
                if info[0] in top_relations:
                    try:
                        nodes = get_related_nodes(info[1], info[0])
                    except Exception:
                        logger.exception("Neighbor query failed for %s", info[1])
                        continue
                    for node in nodes:
                        new_batch.append(Node(name=node[1], id=node[2], clue=top_clue, label=node[0], relations={node[4]: node[3]}))
        return (new_batch, {'input': input_token, 'output': output_token, 'total': total_token})

    def intersection(self, batches, next_clue):
        """Find shared neighbors ignoring direction, with 5,000/10,000 degree cutoffs."""
        new_batch = []
        relations_return = set()
        batch_ids = [set((node.id for node in batch)) for batch in batches]
        if next_clue in self.init_clue2node.keys():
            related_nodes_in_batches = []
            init_node = self.init_clue2node[next_clue]
            related_nodes = get_related_nodes_by_id(init_node.id)
            for batch_id in batch_ids:
                related_nodes_in_batch = [node for node in related_nodes if node[2] in batch_id]
                if related_nodes_in_batch == []:
                    return (None, None)
                related_nodes_in_batches.append(related_nodes_in_batch)
            new_batch.append(Node(name=init_node.name, id=init_node.id, clue=init_node.clue, relations={node[2]: node[0] for nodes in related_nodes_in_batches for node in nodes}))
            return (new_batch, None)
        # Start with the smallest neighborhood to limit intersection work.
        base_batch = batches[0]
        min_num = 10000
        for batch in batches:
            entities_num = 0
            for node in batch:
                entities_num += int(count_nodes(node.id))
            if entities_num < min_num:
                min_num = entities_num
                base_batch = batch
        # Large neighborhoods trigger the semantic fallback in explore().
        if min_num > 5000:
            return (None, None)
        # Cache is local to this intersection, not shared across questions.
        related_nodes_cache = {}

        def get_related_nodes_cached(id, name, related_id):
            if id not in related_nodes_cache:
                related_nodes = get_related_nodes_by_id(id)
                related_nodes_filtered = [related_node for related_node in related_nodes if related_node[2] not in related_id]
                rst = []
                for r_node in related_nodes_filtered:
                    if r_node[1] == 'UnName_Entity':
                        r_related_nodes = get_related_nodes_by_id(r_node[2])
                        r_related_nodes = [r_related_node for r_related_node in r_related_nodes if r_related_node[2] not in [id]]
                        rst.extend(r_related_nodes)
                    else:
                        rst.append(r_node)
                related_nodes_cache[id] = rst
            return related_nodes_cache[id]
        related_nodes = set()
        for node in base_batch:
            for related_node in get_related_nodes_cached(node.id, node.name, node.related_id):
                related_nodes.add((related_node[1], related_node[2]))
        for node in related_nodes:
            entities_num = 0
            entities_num += int(count_nodes(node[1]))
            if entities_num > 10000:
                return (None, None)
        for related_node in related_nodes:
            nodes = get_related_nodes_cached(related_node[1], related_node[0], [])
            matched_nodes = {}
            # Store the first connecting edge for each batch.
            is_candidate = True
            for id_set in batch_ids:
                match = next((node for node in nodes if node[2] in id_set), None)
                if match is None:
                    matched_nodes.clear()
                    is_candidate = False
                    break
                matched_nodes[match[2]] = match[0]
            if is_candidate:
                for id, r in matched_nodes.items():
                    relations_return.add((r, id, related_node[1]))
                new_node = Node(name=related_node[0], id=related_node[1], clue=next_clue, relations=matched_nodes)
                new_batch.append(new_node)
        return (new_batch, relations_return)

def add_tokens(input_tok, output_tok, total_tok, token_num):
    input_tok += token_num['input']
    output_tok += token_num['output']
    total_tok += token_num['total']
    return (input_tok, output_tok, total_tok)


def run_question(data, max_answer_triples=MAX_ANSWER_TRIPLES,
                 allow_own_knowledge=ALLOW_OWN_KNOWLEDGE):
    """Run one question and return its answer, evidence and usage statistics."""
    started = time.perf_counter()
    question = data.get('RawQuestion', data.get('question'))
    if not isinstance(question, str) or not question:
        raise ValueError('Expected RawQuestion (WebQSP) or question (CWQ).')
    result_triplets = []
    cot_answer = False
    # Construct the question graph, then anchor its specific entity clues in KG.
    statement, specific, generic, usage = complete_info(question)
    input_token, output_token, total_token = add_tokens(0, 0, 0, usage)
    triples, target, usage = structure(statement, generic, specific)
    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, usage)
    llm_call = 2
    clues = list(dict.fromkeys(clue for head, _, tail in triples for clue in (head, tail)))
    pairs = [(head, tail) for head, _, tail in triples]
    initial_nodes = []
    for clue in clues:
        if clue not in specific:
            continue
        name = clue[1:-1] if re.match(r'^["\'].*["\']$', clue) else clue
        nodes = query_node_by_name(name)
        if len(nodes) == 1:
            initial_nodes.append(Node(nodes[0][0], nodes[0][1], clue))
        elif len(nodes) > 1:
            # Keep dataset-provided topic-entity disambiguation.
            topics = data.get('topic_entity', {})
            for name, node_id in nodes:
                if node_id in topics:
                    initial_nodes.append(Node(name, node_id, clue))
                    break
    if initial_nodes:
        exploration = Exploration(initial_nodes, pairs, triples, target, generic, statement)
        while not exploration.exceed:
            if exploration.chain.next_clue is None:
                if target in exploration.chain.clues:
                    break
                # Restart exploration from the remaining anchors in insertion order.
                for clue in exploration.init_clue2node:
                    if clue not in exploration.chain.clues and clue not in exploration.explored_init:
                        exploration.chain = NodeChain(exploration.init_clue2node[clue], pairs)
                        exploration.explored_init.append(clue)
                if exploration.chain.next_clue is None:
                    break
                continue
            batch, usage = exploration.explore()
            input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, usage)
            exploration.chain.add_batch(batch)
        llm_call += exploration.llm_call
        # Supply retrieved evidence to the final answer-generation prompt.
        result_triplets = exploration.chain.generate_triples()
        if max_answer_triples is not None:
            result_triplets = result_triplets[:max_answer_triples]
        answer, usage = answer_question(question, result_triplets, allow_own_knowledge)
    elif allow_own_knowledge:
        answer, usage = chat('please answer the question step by step: ' + question)
        cot_answer = True
    else:
        answer, usage = answer_question(question, [], allow_own_knowledge=False)
    input_token, output_token, total_token = add_tokens(input_token, output_token, total_token, usage)
    return {
        'data': data, 'structure': triples, 'answer_w_triples': answer,
        'llm_call': llm_call + 1, 'triples': result_triplets,
        'input_token': input_token, 'output_token': output_token,
        'total_token': total_token, 'time_used': time.perf_counter() - started,
        'cot_answer': cot_answer,
    }


def run_dataset(data_path=DATA_PATH, output_path=OUTPUT_PATH,
                sample_limit=SAMPLE_LIMIT, max_answer_triples=MAX_ANSWER_TRIPLES,
                allow_own_knowledge=ALLOW_OWN_KNOWLEDGE):
    """Write JSONL with explicit failures; use a new output file for each run."""
    for value in (sample_limit, max_answer_triples):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError('Limits must be nonnegative integers or None.')
    dataset = prepare_dataset(data_path)
    if sample_limit is not None:
        dataset = dataset[:sample_limit]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    succeeded = failed = 0
    # Keep each run in a separate output file.
    with output_path.open('x', encoding='utf-8') as handle:
        for index, data in enumerate(dataset):
            logger.info('Question %s/%s', index + 1, len(dataset))
            try:
                result = run_question(data, max_answer_triples, allow_own_knowledge)
            except Exception as exc:
                logger.exception('Question %s failed', index + 1)
                result = {'data': data, 'error': str(exc), 'error_type': type(exc).__name__}
                failed += 1
            else:
                succeeded += 1
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
    logger.info('Saved %s: %s completed, %s failed', output_path, succeeded, failed)
    return succeeded, failed


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    configure()
    run_dataset()
