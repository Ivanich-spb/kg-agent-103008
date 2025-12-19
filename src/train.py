"""
Training script to fine-tune a base LLM on code-based instructions for KG-Agent.

This is a skeleton that demonstrates structure: dataset loading, model init,
optimizer, and simple training loop. Replace with HuggingFace Trainer or
a JAX-based loop as needed.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader, Dataset

# TODO: replace with transformers dataset/tokenizer/model imports


class InstructionDataset(Dataset):
    """A small Dataset wrapper for code-instruction items."""

    def __init__(self, items: list[Dict[str, str]]) -> None:
        super().__init__()
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


def train_loop(model: Any, dataloader: DataLoader, device: torch.device, epochs: int = 1) -> None:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            # TODO: tokenize prompts and compute loss using LM head
            loss = torch.tensor(0.0, device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1} completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=False, default='data/instructions.json')
    parser.add_argument('--output', type=str, required=False, default='outputs')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # TODO: load synthesized instruction dataset
    items = []  # placeholder: list of {prompt, code_plan, answer}
    dataset = InstructionDataset(items)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # TODO: initialize a small LLM or load a pretrained model (e.g., LLaMA via transformers)
    # For skeleton we create a tiny torch Module
    class DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, *args, **kwargs):
            return {}

    model = DummyLM()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loop(model, loader, device, epochs=args.epochs)

    # save checkpoint
    ckpt_path = os.path.join(args.output, 'checkpoint.pt')
    torch.save({'model_state': model.state_dict()}, ckpt_path)
    print('Saved checkpoint to', ckpt_path)


if __name__ == '__main__':
    main()
