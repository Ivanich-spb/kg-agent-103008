# KG-Agent

Paper: KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph
Link: http://arxiv.org/abs/2402.11163v1

Overview
--------
This repository is a minimal skeleton implementation for the KG-Agent paper. KG-Agent proposes an autonomous LLM-based agent that interacts with a knowledge graph (KG) via a multifunctional toolbox, a KG-based executor, and a knowledge memory. The agent iteratively selects tools, executes KG operations, and updates memory to perform multi-hop reasoning. The paper also describes using code-formatted instructions to fine-tune a base LLM (e.g., LLaMA-7B) with a small amount of data.

Repository structure
--------------------
- README.md            - this file
- requirements.txt     - python dependencies
- Dockerfile           - container for running experiments
- src/model.py         - core KG-Agent model skeleton
- src/data.py          - KG and dataset utilities, synthetic instruction generator
- src/train.py         - training / fine-tuning script (skeleton)
- src/evaluate.py      - evaluation script (skeleton)

Quickstart
----------
1. Create a virtualenv (Python 3.10+)
2. pip install -r requirements.txt
3. Prepare a KG (edge list or RDF) and a dataset (QA pairs)
4. Run training (skeleton):
   python src/train.py --data data/ --output outputs/
5. Evaluate (skeleton):
   python src/evaluate.py --checkpoint outputs/checkpoint.pt --data data/

Notes
-----
- This repository contains skeleton code and interfaces. Significant implementation is required to reproduce the results in the paper (LLM fine-tuning, KG executor, evaluation datasets).
- TODOs are marked throughout the code.

License
-------
MIT (placeholder)
