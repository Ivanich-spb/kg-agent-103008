"""
Data utilities: load KG, load datasets, and synthesize code-based instructions.
"""

from typing import Any, Dict, Iterable, List, Tuple
import networkx as nx
import json


def load_kg_from_edgelist(path: str):
    """Load a KG from a TSV/CSV edge list with (head, relation, tail) per line.

    Returns a directed NetworkX graph where edges carry relation attribute.
    """
    G = nx.DiGraph()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            G.add_node(h)
            G.add_node(t)
            G.add_edge(h, t, relation=r)
    return G


def load_qa_pairs(path: str):
    """Load question-answer pairs from a JSONL or JSON file.

    Expected format: [{"question": "...", "answer": "..."}, ...]
    """
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data


def synthesize_code_instructions(qa_pairs: Iterable[Tuple[str, str]], max_samples: int = 10000):
    """Synthesize a code-style instruction dataset to fine-tune the LLM.

    Each item is a dict {"prompt": ..., "code_plan": ..., "answer": ...}

    This is a stub: real synthesis requires careful template design and KG-aware transformations.
    """
    out = []
    for i, (q, a) in enumerate(qa_pairs):
        if i >= max_samples:
            break
        prompt = f"# Question\n{q}\n# Generate a step-by-step program to query the KG and derive the answer."
        code_plan = f"# pseudo-code plan for: {q}"  # TODO: implement templates
        out.append({"prompt": prompt, "code_plan": code_plan, "answer": a})
    return out


if __name__ == "__main__":
    # smoke test for data utilities
    G = nx.DiGraph()
    G.add_edge('A', 'B', relation='rel')
    print('Nodes:', list(G.nodes))
