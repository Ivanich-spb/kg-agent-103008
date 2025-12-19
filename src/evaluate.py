"""
Evaluation script to run the KG-Agent on QA datasets and compute metrics.

This script is a high-level scaffold: it loads a model checkpoint, runs the
agent on questions, and computes simple metrics like accuracy.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Dict
import torch

# Note: this file assumes that src.model.KGAgent exists and can be instantiated
from src.model import KGAgent


def load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location='cpu')


def evaluate(agent: KGAgent, qa_pairs: List[Dict[str, str]]) -> Dict[str, float]:
    total = 0
    correct = 0
    for item in qa_pairs:
        q = item.get('question', '')
        gold = item.get('answer', '').strip().lower()
        pred = agent.run_episode(q).strip().lower()
        total += 1
        if pred and gold and pred == gold:
            correct += 1
    acc = correct / total if total > 0 else 0.0
    return {'accuracy': acc, 'total': total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    # TODO: construct KGAgent from checkpoint (weights, config)
    agent = KGAgent(llm=None)

    with open(args.data, 'r', encoding='utf-8') as fh:
        qa_pairs = json.load(fh)

    metrics = evaluate(agent, qa_pairs)
    print('Evaluation results:', metrics)


if __name__ == '__main__':
    main()
