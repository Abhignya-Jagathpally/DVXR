"""dvxr.llm.finetune — the FULL Option 3: trainable soft-prompt + LoRA-adapted last block
+ classification head, finetuned through an otherwise-frozen LLM.

The frozen-probe predictor in ``predictor.py`` leaves gains on the table (rasbt ch6: a
head-only probe scores ~72% where unfreezing the last transformer block scores ~95%). Here
we apply the rasbt classification-finetuning recipe to our SOFT-PROMPT model:

  * per-modality soft-prompt projections + a classification head are trained;
  * the last ``lora_blocks`` transformer blocks get **LoRA** adapters (rank r, alpha) on
    their attention/MLP linears — trainable, while the base weights stay frozen;
  * the **last-token** hidden state is pooled (rasbt: best for a causal LM), cross-entropy,
    AdamW.

LoRA (not raw unfreezing) is deliberate: the reader's LLM is module-cached and SHARED across
CV folds, so unfreezing its weights would leak across folds. LoRA adapters are created fresh
per call and removed afterwards, so the base model is never mutated — leakage-free and
memory-light (rasbt: LoRA matches full finetuning with less overfitting — right for tiny
clinical N). Trained inside each fold on train indices only; import-guarded (torch).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from dvxr.llm.predictor import (
    SOFT_PROMPT_PREFIX,
    _modality_quant,
    _seeded_matrix,
    get_reader,
)


# ------------------------------------------------------------------ LoRA (rasbt appendix E)
def _make_lora_linear(linear, rank: int, alpha: int, seed: int):
    import torch
    from torch import nn

    class LinearWithLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear                       # frozen base
            torch.manual_seed(seed)
            self.A = nn.Parameter(torch.empty(linear.in_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(rank, linear.out_features))  # B=0 -> starts no-op
            self.scale = alpha / rank

        def forward(self, x):
            return self.linear(x) + self.scale * (x.to(self.A.dtype) @ self.A @ self.B).to(x.dtype)

    return LinearWithLoRA()


def _inject_lora(block, rank: int, alpha: int, seed: int) -> List[tuple]:
    """Wrap every nn.Linear in ``block`` with LoRA. Returns [(parent, attr, original)] so it
    can be UNDONE afterwards (the shared base model must be left pristine). The Linear
    children are SNAPSHOT before mutating so the walk never descends into new LoRA wrappers."""
    import torch
    from torch import nn

    targets = []  # (parent, attr, child) snapshot
    for parent in block.modules():
        for attr, child in parent.named_children():
            if isinstance(child, nn.Linear):
                targets.append((parent, attr, child))
    undo: List[tuple] = []
    for i, (parent, attr, child) in enumerate(targets):
        setattr(parent, attr, _make_lora_linear(child, rank, alpha, seed + i))
        undo.append((parent, attr, child))
    return undo


def _remove_lora(undo: List[tuple]) -> None:
    for parent, attr, original in undo:
        setattr(parent, attr, original)


# ------------------------------------------------------------------ classifier
def _build_classifier(reader, modalities, d_code, n_classes, seed, pool, lora_undo):
    import torch
    from torch import nn

    hidden = reader.hidden

    class SoftPromptClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.modalities = list(modalities)
            self.pool = pool
            self.proj = nn.ParameterDict()
            self.absent = nn.ParameterDict()
            for m in modalities:
                self.proj[m] = nn.Parameter(
                    torch.tensor(_seeded_matrix(hidden, d_code, seed, f"proj:{m}")))
                self.absent[m] = nn.Parameter(
                    torch.tensor(_seeded_matrix(1, hidden, seed, f"absent:{m}")[0]))
            self.head = nn.Linear(hidden, n_classes)
            tok = reader._tok(SOFT_PROMPT_PREFIX, return_tensors="pt").input_ids.to(reader.device)
            with torch.no_grad():
                self.register_buffer("prompt_emb",
                                     reader._model.get_input_embeddings()(tok)[0])

        def forward(self, quant_by_mod, present):
            n = len(next(iter(quant_by_mod.values())))
            cols = []
            for m in self.modalities:
                if present.get(m, True):
                    cols.append((quant_by_mod[m] @ self.proj[m].T).unsqueeze(1))
                else:
                    cols.append(self.absent[m].expand(n, 1, -1))
            soft = torch.cat(cols, dim=1)
            pe = self.prompt_emb.unsqueeze(0).expand(n, -1, -1)
            inp = torch.cat([soft, pe], dim=1)
            attn = torch.ones(inp.shape[:2], dtype=torch.long, device=inp.device)
            h = reader._model(inputs_embeds=inp, attention_mask=attn).hidden_states[-1]
            pooled = h[:, -1, :] if self.pool == "last" else h.mean(dim=1)
            return self.head(pooled)

    return SoftPromptClassifier()


def finetune_softprompt(task, tr, te, seed: int = 7, d_code: int = 24,
                        epochs: int = 20, lr: float = 5e-3, pool: str = "last",
                        lora_blocks: int = 1, lora_rank: int = 8, lora_alpha: int = 16,
                        drop: Optional[List[str]] = None) -> np.ndarray:
    """Finetune soft-prompt + LoRA(last ``lora_blocks`` blocks) + head on TRAIN; return
    test-fold P(class=1). Full-batch (CPU-friendly). ``drop`` = missing modalities.
    LoRA adapters are injected fresh and removed after — the shared base LLM is untouched."""
    import torch
    from torch import nn

    reader = get_reader(d_code=d_code, seed=seed)
    quant_np = _modality_quant(task, seed, d_code)
    device = reader.device
    dtype = next(reader._model.parameters()).dtype
    quant = {m: torch.tensor(v, dtype=dtype, device=device) for m, v in quant_np.items()}
    present = {m: (drop is None or m not in drop) for m in task.modalities}

    y = np.asarray(task.y, dtype=int)
    classes = sorted(np.unique(y[tr]).tolist())
    if len(classes) < 2:
        return np.full(len(te), float(classes[0]))

    # inject LoRA into the last `lora_blocks` transformer layers (Qwen: model.model.layers)
    layers = reader._model.model.layers
    undo: List[tuple] = []
    for li in range(max(0, len(layers) - lora_blocks), len(layers)):
        undo += _inject_lora(layers[li], lora_rank, lora_alpha, seed + 100 * li)
    try:
        clf = _build_classifier(reader, task.modalities, d_code, len(classes),
                                seed, pool, undo).to(device)
        # base LLM is fully frozen, so any requires_grad param inside it is a fresh LoRA
        # adapter; the classifier owns the soft-prompt projections + absent tokens + head.
        lora_ps = [p for p in reader._model.parameters() if p.requires_grad]
        trainable = [p for p in clf.parameters() if p.requires_grad] + lora_ps
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.1)
        # class-weighted CE (clinical labels are imbalanced; rasbt balances by resampling)
        cls_index = {c: i for i, c in enumerate(classes)}
        counts = np.array([(y[tr] == c).sum() for c in classes], dtype=float)
        w = torch.tensor((counts.sum() / (len(classes) * np.clip(counts, 1, None))),
                         dtype=torch.float32, device=device)
        lossf = nn.CrossEntropyLoss(weight=w)
        tr = np.asarray(tr)
        ytr = torch.tensor([cls_index[int(v)] for v in y[tr]], device=device)
        qb_tr = {m: quant[m][tr] for m in task.modalities}

        clf.train()
        for _ in range(epochs):                       # full-batch (few, expensive CPU passes)
            logits = clf(qb_tr, present)
            loss = lossf(logits, ytr)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        clf.eval()
        te = np.asarray(te)
        pos = cls_index.get(1, len(classes) - 1)
        with torch.no_grad():
            qb_te = {m: quant[m][te] for m in task.modalities}
            prob = torch.softmax(clf(qb_te, present), dim=1)[:, pos].float().cpu().numpy()
        return prob
    finally:
        _remove_lora(undo)                            # leave the shared base LLM pristine
