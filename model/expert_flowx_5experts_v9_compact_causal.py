# -*- coding: utf-8 -*-
"""
5-expert v9 compact answers: Stage3 causal teacher -> CausalGate -> Stage4.3
=================================================================

This file is intentionally an extension layer on top of:
  expert_flowx_5experts_v9_compact.py

Uses only the matching v9 Stage0/1/2 checkpoints (legacy weights require retraining):
  Stage 3
    - neutral-factor counterfactual teacher over 5 experts
    - learned CausalGate
    - optional Stage3 cross-attention bias training
    - SHA256 provenance guard for main.pt / flowx.pt / teacher cache

  Stage 4.3
    - starts DIRECTLY from Stage1 main.pt
    - freezes experts, FlowX and Stage3 CausalGate
    - trains decoder cross-attention only
    - actual Attention x Value -> out_proj contribution distillation
    - distractor-output invariance
    - target-contribution direction consistency
    - random 1-2 distractor experts per single-factor question
    - NO Stage3 bias in the Stage4.3 student forward

The 5 experts are:
  shape / color / size / pattern / spacing

Important experimental rule:
  main.pt and flowx.pt must be the SAME lineage used to train stage3.pt.
  The script writes stage3_provenance.json and refuses Stage4.3 if hashes do not match.
"""

from __future__ import annotations

import os
import sys
import math
import json
import random
import hashlib
import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# Base-model dynamic import
# =====================================================================

def load_base_module(path: str):
    path = str(Path(path).resolve())
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Base 5-expert model was not found: {path}\n"
            "Pass --base-model with the exact path to expert_flowx_5experts_v9_compact.py"
        )
    spec = importlib.util.spec_from_file_location("factor_v9_5expert_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base model: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# Stage3/4.3 config (separate from base CFG so current checkpoints remain unchanged)
# =====================================================================
@dataclass
class ExtraCFG:
    # Stage3
    epochs_stage3_warmup: int = 15
    epochs_stage3_joint: int = 35
    stage3_lr: float = 1e-4
    stage3_causal_temp: float = 1.0
    stage3_effect_clip: float = 12.0
    stage3_gate_lambda: float = 1.0
    stage3_answer_lambda: float = 0.50
    stage3_bias_reg: float = 1e-3
    stage3_affinity_init: float = 0.15
    stage3_gate_init: float = 0.35
    # Main-question teacher can encode family shortcuts; keep it weaker than factor Qs.
    stage3_main_gate_scale: float = 0.0

    # Stage4.3
    epochs_stage43: int = 30
    stage43_lr: float = 2e-5
    stage43_answer_lambda: float = 1.0
    stage43_clean_contrib_lambda: float = 0.02
    # Main question is intentionally downweighted so A-vs-B family exclusivity is not over-imposed.
    stage43_main_contrib_scale: float = 0.0
    stage43_pert_answer_lambda: float = 0.50
    stage43_pert_contrib_lambda: float = 0.02
    stage43_output_invariance_lambda: float = 0.10
    stage43_direction_lambda: float = 0.02
    stage43_anchor_lambda: float = 2e-3
    stage43_grad_clip: float = 1.0
    stage43_warmup_epochs: int = 5
    stage43_perturb_min: float = 0.05
    stage43_perturb_max: float = 0.20
    stage43_min_distractors: int = 1
    stage43_max_distractors: int = 2
    stage43_seed: int = 4305


# =====================================================================
# Contribution-aware decoder. State dict is checkpoint-compatible with base MainFlow.
# =====================================================================
class CausalMainDecoderLayer(nn.Module):
    def __init__(self, base_cfg):
        super().__init__()
        self.cfg = base_cfg
        self.self_attn = nn.MultiheadAttention(base_cfg.d_model, base_cfg.n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(base_cfg.d_model, base_cfg.n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(base_cfg.d_model, base_cfg.d_model * 4),
            nn.GELU(),
            nn.Linear(base_cfg.d_model * 4, base_cfg.d_model),
        )
        self.n1, self.n2, self.n3 = (nn.LayerNorm(base_cfg.d_model) for _ in range(3))

    def _expert_value_contributions(self, z: torch.Tensor, attn_w: torch.Tensor):
        """Reconstruct expert-specific effective cross-attention output contribution.

        C_i = W_O( concat_h sum_{k in expert_i} alpha[h,q,k] * V[h,k] )

        out_proj.bias is excluded because it is shared by all experts.
        Return: (B,Lq,n_experts,D)
        """
        mha = self.cross_attn
        B, Lk, D = z.shape
        if attn_w.dim() != 4:
            raise ValueError(f"expected attn_w=(B,H,Lq,Lk), got {tuple(attn_w.shape)}")
        Ba, H, Lq, Lka = attn_w.shape
        if Ba != B or Lka != Lk:
            raise ValueError(f"attention/value mismatch: attn={tuple(attn_w.shape)} z={tuple(z.shape)}")
        head_dim = D // H

        if mha.in_proj_weight is not None:
            w_v = mha.in_proj_weight[2 * D: 3 * D]
            b_v = None if mha.in_proj_bias is None else mha.in_proj_bias[2 * D: 3 * D]
        else:
            w_v = mha.v_proj_weight
            b_v = None if mha.in_proj_bias is None else mha.in_proj_bias[2 * D: 3 * D]

        v = F.linear(z, w_v, b_v)
        v = v.view(B, Lk, H, head_dim).transpose(1, 2)  # B,H,Lk,Dh

        n = len(self.cfg.experts)
        cl = self.cfg.compressed_len
        if Lk != n * cl:
            raise ValueError(f"expected Lk={n*cl}, got {Lk}")

        parts = []
        for i in range(n):
            s, e = i * cl, (i + 1) * cl
            ctx_h = torch.einsum("bhqk,bhkd->bhqd", attn_w[..., s:e], v[:, :, s:e, :])
            ctx = ctx_h.transpose(1, 2).contiguous().view(B, Lq, D)
            c_i = F.linear(ctx, mha.out_proj.weight, bias=None)
            parts.append(c_i)
        return torch.stack(parts, dim=2)

    def forward(self, y, z, causal_mask, cross_bias=None,
                return_cross_attn=False, return_cross_contrib=False):
        h, _ = self.self_attn(y, y, y, attn_mask=causal_mask, need_weights=False)
        y = self.n1(y + h)

        need_w = bool(return_cross_attn or return_cross_contrib or cross_bias is not None)
        if not need_w:
            h, attn_w = self.cross_attn(y, z, z, need_weights=False)
        else:
            h, attn_w = self.cross_attn(
                y, z, z,
                attn_mask=(cross_bias.contiguous() if cross_bias is not None else None),
                need_weights=True,
                average_attn_weights=False,
            )

        contrib = None
        if return_cross_contrib:
            contrib = self._expert_value_contributions(z, attn_w)

        y = self.n2(y + h)
        y = self.n3(y + self.ff(y))

        if return_cross_contrib:
            return y, attn_w, contrib
        if return_cross_attn:
            return y, attn_w
        return y


class CausalMainFlow(nn.Module):
    """Same parameter names/shapes as base.MainFlow, plus contribution-return path."""
    def __init__(self, base, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleDict({
            name: base.Expert(cfg, vocab_size, mode=cfg.expert_transforms.get(name, "none"))
            for name in cfg.experts
        })
        self.ans_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.ans_pos = nn.Parameter(torch.randn(1, cfg.max_answer_len, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList([CausalMainDecoderLayer(cfg) for _ in range(cfg.dec_layers)])
        self.out = nn.Linear(cfg.d_model, vocab_size)

    def build_Z(self, img, q):
        sms = []
        for name in self.cfg.experts:
            _, sm = self.experts[name](img, q)
            sms.append(sm)
        return torch.cat(sms, dim=1), sms

    def decode_from_Z(self, z, ans_in, cross_bias=None,
                      return_cross_attn=False, return_cross_contrib=False):
        L = ans_in.size(1)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=ans_in.device), 1)
        y = self.ans_emb(ans_in) + self.ans_pos[:, :L]
        attns, contribs = [], []
        for lyr in self.layers:
            if return_cross_contrib:
                y, aw, cc = lyr(
                    y, z, causal, cross_bias=cross_bias,
                    return_cross_attn=True, return_cross_contrib=True,
                )
                attns.append(aw)
                contribs.append(cc)
            elif return_cross_attn:
                y, aw = lyr(y, z, causal, cross_bias=cross_bias, return_cross_attn=True)
                attns.append(aw)
            else:
                y = lyr(y, z, causal, cross_bias=cross_bias)
        logits = self.out(y)
        if return_cross_contrib:
            return logits, attns, contribs
        if return_cross_attn:
            return logits, attns
        return logits

    def forward(self, img, q, ans_in, cross_bias=None,
                return_cross_attn=False, return_cross_contrib=False):
        z, _ = self.build_Z(img, q)
        return self.decode_from_Z(
            z, ans_in, cross_bias=cross_bias,
            return_cross_attn=return_cross_attn,
            return_cross_contrib=return_cross_contrib,
        )


# =====================================================================
# Generic helpers
# =====================================================================
def ce_loss(logits, target, pad_id):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_id,
    )


def flowx_hidden(flowx, z):
    # Current v4 Stage2 FlowXGate exposes encode(z).
    if hasattr(flowx, "encode"):
        return flowx.encode(z)
    out = flowx(z)
    if isinstance(out, tuple):
        return out[-1]
    return out


def _state_dict_sha256(module: nn.Module) -> str:
    """Stable SHA256 for a module state_dict, including 0-D scalar tensors."""
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        t = t.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        # 0-D scalar tensors (e.g. Stage3 raw_*_scale) cannot be dtype-viewed
        # directly from float32 -> uint8. Flatten first so they become length-1.
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    return math.log(math.expm1(x))


def _expand_cross_bias(bias_row: torch.Tensor, query_len: int, n_heads: int):
    B, _, Lk = bias_row.shape
    b = bias_row.expand(B, query_len, Lk)
    b = b.unsqueeze(1).expand(B, n_heads, query_len, Lk)
    return b.reshape(B * n_heads, query_len, Lk).contiguous()


def question_type_from_q(base, q_row: torch.Tensor, vocab, cfg) -> str:
    text = vocab.decode(q_row.detach().cpu().tolist())
    return base.question_type(text, cfg)


def sample_type(base, prompt: str, label: str, cfg) -> str:
    qt = base.question_type(prompt, cfg)
    if qt == "main":
        if not all(base.parse_structured_answer(label).values()):
            raise ValueError(f"Invalid v9 joint-main sentence: {label!r}")
        return "main_union"
    return qt


# =====================================================================
# Stage3: neutral intervention teacher
# =====================================================================
@torch.no_grad()
def target_content_logprob_batch_from_Z(model, z, a, vocab):
    ans_in = a[:, :-1]
    target = a[:, 1:]
    logits = model.decode_from_Z(z, ans_in)
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    pad_id = vocab.stoi[vocab.PAD]
    eos_id = vocab.stoi[vocab.EOS]
    mask = content_query_mask(a, vocab)
    denom = mask.sum(dim=1).clamp_min(1)
    sums = (tok_lp * mask.to(tok_lp.dtype)).sum(dim=1)

    empty = mask.sum(dim=1) == 0
    if empty.any():
        mask2 = target != pad_id
        denom2 = mask2.sum(dim=1).clamp_min(1)
        sums2 = (tok_lp * mask2.to(tok_lp.dtype)).sum(dim=1)
        sums = torch.where(empty, sums2, sums)
        denom = torch.where(empty, denom2, denom)
    return sums / denom


@torch.no_grad()
def neutral_factor_effects_batch(model, vocab, img, q, a, cfg):
    model.eval()
    z, sms = model.build_Z(img, q)
    lp_full = target_content_logprob_batch_from_Z(model, z, a, vocab)

    neutral_img = torch.full_like(img, -1.0)
    effects = []
    for i, name in enumerate(cfg.experts):
        _, sm_null = model.experts[name](neutral_img, q)
        cf_sms = list(sms)
        cf_sms[i] = sm_null
        z_cf = torch.cat(cf_sms, dim=1)
        lp_cf = target_content_logprob_batch_from_Z(model, z_cf, a, vocab)
        effects.append(lp_full - lp_cf)
    return z, torch.stack(effects, dim=-1)


class IndexedDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, q, a = self.base[i]
        return img, q, a, i


def causal_target_probs(effects, extra: ExtraCFG):
    e = effects.clamp(-extra.stage3_effect_clip, extra.stage3_effect_clip)
    return F.softmax(e / max(extra.stage3_causal_temp, 1e-6), dim=-1)


class Stage3CausalBias(nn.Module):
    def __init__(self, cfg, extra: ExtraCFG):
        super().__init__()
        self.cfg = cfg
        self.extra = extra
        n = len(cfg.experts)
        d = cfg.d_model
        hidden = d * 2
        self.gate_net = nn.Sequential(
            nn.Linear((n + 1) * d, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d),
            nn.GELU(),
            nn.Linear(d, n),
        )
        self.raw_affinity_scale = nn.Parameter(
            torch.tensor(_inv_softplus(extra.stage3_affinity_init), dtype=torch.float32)
        )
        self.raw_gate_scale = nn.Parameter(
            torch.tensor(_inv_softplus(extra.stage3_gate_init), dtype=torch.float32)
        )

    def gate_logits(self, z, h):
        B, L, D = z.shape
        n = len(self.cfg.experts)
        cl = self.cfg.compressed_len
        if L != n * cl:
            raise ValueError(f"Stage3 expects Z length {n*cl}, got {L}")
        blocks = z.view(B, n, cl, D).mean(dim=2)
        x = torch.cat([blocks.flatten(1), h], dim=-1)
        return self.gate_net(x)

    def gate_probs(self, z, h):
        return F.softmax(self.gate_logits(z, h), dim=-1)

    def bias_row(self, z, h, gate_logits=None):
        B, L, D = z.shape
        n = len(self.cfg.experts)
        cl = self.cfg.compressed_len
        if gate_logits is None:
            gate_logits = self.gate_logits(z, h)
        gate = F.softmax(gate_logits, dim=-1)
        token_gate = gate.repeat_interleave(cl, dim=-1).unsqueeze(1)

        aff = (h.unsqueeze(1) @ z.transpose(1, 2)) / math.sqrt(D)
        aff = (aff - aff.mean(dim=-1, keepdim=True)) / (
            aff.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        )
        gate_rel = token_gate * n
        gate_log = torch.log(gate_rel.clamp_min(1e-6))
        affinity_scale = F.softplus(self.raw_affinity_scale)
        gate_scale = F.softplus(self.raw_gate_scale)
        return affinity_scale * aff * gate_rel + gate_scale * gate_log


def _teacher_cache_path(ckpt):
    return Path(ckpt) / "stage3_causal_teacher.pt"


def _provenance_path(ckpt):
    return Path(ckpt) / "stage3_provenance.json"


@torch.no_grad()
def precompute_stage3_teacher(base, model, vocab, dataset, cfg, extra, cache_path, rebuild=False):
    cache_path = Path(cache_path)
    current_main_hash = _state_dict_sha256(model)
    teacher_signature = hashlib.sha256(json.dumps({
        "train":base.training_fingerprint(cfg),"vocab":vocab.itos,
        "items":[(p.name,q,a) for p,q,a in dataset.items],
        "architecture":{k:getattr(cfg,k) for k in base.ARCH_KEYS},
        "temp":extra.stage3_causal_temp,"clip":extra.stage3_effect_clip,
        "method":"neutral-value-only-v9-compact"
    },sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if cache_path.exists() and not rebuild:
        obj = torch.load(cache_path, map_location="cpu")
        if (
            obj.get("version") == 9
            and obj.get("teacher_signature") == teacher_signature
            and obj.get("n") == len(dataset)
            and tuple(obj.get("experts", ())) == tuple(cfg.experts)
            and obj.get("main_state_sha256") == current_main_hash
        ):
            print(f"[Stage3 teacher] cache loaded + main hash verified: {cache_path}")
            print_stage3_teacher_summary(base, dataset, obj, cfg)
            return obj
        print("[Stage3 teacher] cache/main provenance mismatch -> rebuild")

    loader = DataLoader(IndexedDataset(dataset), batch_size=cfg.batch, shuffle=False, drop_last=False)
    N = len(dataset)
    n_exp = len(cfg.experts)
    effects_all = torch.zeros(N, n_exp, dtype=torch.float32)

    print("[Stage3 teacher] precomputing 5-expert neutral-factor counterfactual effects ...")
    for bi, (img, q, a, idx) in enumerate(loader):
        img, q, a = img.to(cfg.device), q.to(cfg.device), a.to(cfg.device)
        _, eff = neutral_factor_effects_batch(model, vocab, img, q, a, cfg)
        effects_all[idx.long()] = eff.detach().cpu()
        if (bi + 1) % 10 == 0 or bi + 1 == len(loader):
            print(f"  batch {bi+1}/{len(loader)}")

    obj = {
        "version": 9,
        "teacher_signature": teacher_signature,
        "n": N,
        "experts": tuple(cfg.experts),
        "main_state_sha256": current_main_hash,
        "effects": effects_all,
        "probs": causal_target_probs(effects_all, extra),
    }
    torch.save(obj, cache_path)
    print(f"[Stage3 teacher] saved: {cache_path}")
    print_stage3_teacher_summary(base, dataset, obj, cfg)
    return obj


def print_stage3_teacher_summary(base, dataset, teacher, cfg):
    effects = teacher["effects"]
    groups: Dict[str, List[torch.Tensor]] = {}
    hits = total = 0
    expert_names = list(cfg.experts)
    for i in range(len(dataset)):
        _path, prompt, label = dataset.items[i]
        st = sample_type(base, prompt, label, cfg)
        groups.setdefault(st, []).append(effects[i])
        if st in expert_names:
            total += 1
            hits += int(int(torch.argmax(effects[i]).item()) == expert_names.index(st))

    print("[Stage3 teacher neutral-factor summary]")
    order = expert_names + ["main_A", "main_B", "main_union", "main", "mixed"]
    for st in order:
        if st not in groups:
            continue
        m = torch.stack(groups[st]).mean(dim=0)
        txt = " ".join(f"{name}={m[j].item():+.4f}" for j, name in enumerate(expert_names))
        print(f"  {st:<10}: n={len(groups[st]):>4} {txt}")
    if total:
        print(f"  teacher argmax routing(factor questions): {hits}/{total} = {hits/total:.2%}")


def _sample_gate_weights(base, dataset, idx: torch.Tensor, cfg, extra, device):
    w = torch.ones(idx.numel(), device=device)
    for j, ii in enumerate(idx.detach().cpu().tolist()):
        _path, prompt, _label = dataset.items[ii]
        qt = base.question_type(prompt, cfg)
        if qt not in cfg.experts:
            w[j] = extra.stage3_main_gate_scale
    return w


def _soft_target_ce_per_sample(logits, target_probs):
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


@torch.no_grad()
def evaluate_stage3_gate_fit(base, model, flowx, controller, teacher, dataset, vocab, cfg, extra):
    loader = DataLoader(IndexedDataset(dataset), batch_size=cfg.batch, shuffle=False, drop_last=False)
    tprobs = teacher["probs"]
    sums = {"ce": 0.0, "kl": 0.0, "n": 0, "agree": 0, "route": 0, "route_n": 0}
    expert_names = list(cfg.experts)
    by = {e: [0, 0] for e in expert_names}

    for img, q, _a, idx in loader:
        img, q = img.to(cfg.device), q.to(cfg.device)
        with torch.no_grad():
            z, _ = model.build_Z(img, q)
            h = flowx_hidden(flowx, z)
            gp = controller.gate_probs(z, h)
        tp = tprobs[idx.long()].to(cfg.device)
        ce = -(tp * gp.clamp_min(1e-8).log()).sum(-1)
        kl = (tp.clamp_min(1e-8) * (tp.clamp_min(1e-8).log() - gp.clamp_min(1e-8).log())).sum(-1)
        sums["ce"] += ce.sum().item()
        sums["kl"] += kl.sum().item()
        sums["n"] += idx.numel()
        sums["agree"] += (tp.argmax(-1) == gp.argmax(-1)).sum().item()

        for b, ii in enumerate(idx.detach().cpu().tolist()):
            _path, prompt, _label = dataset.items[ii]
            qt = base.question_type(prompt, cfg)
            if qt in expert_names:
                pred = expert_names[int(gp[b].argmax().item())]
                sums["route_n"] += 1
                by[qt][1] += 1
                if pred == qt:
                    sums["route"] += 1
                    by[qt][0] += 1

    n = max(1, sums["n"])
    print("[Stage3 gate fit]")
    print(f"  softCE={sums['ce']/n:.6f} KL(T||G)={sums['kl']/n:.6f} teacher-argmax agree={sums['agree']/n:.2%}")
    if sums["route_n"]:
        print(f"  expected-factor routing={sums['route']}/{sums['route_n']}={sums['route']/sums['route_n']:.2%}")
    for e in expert_names:
        ok, nn_ = by[e]
        if nn_:
            print(f"    {e:<8}: {ok}/{nn_}={ok/nn_:.2%}")
    return {
        "soft_ce": sums["ce"] / n,
        "kl_teacher_gate": sums["kl"] / n,
        "teacher_argmax_agreement": sums["agree"] / n,
        "expected_factor_routing": sums["route"] / max(1, sums["route_n"]),
    }


def write_stage3_provenance(ckpt, model, flowx, teacher):
    obj = {
        "version": 2,
        "main_state_sha256": _state_dict_sha256(model),
        "flowx_state_sha256": _state_dict_sha256(flowx),
        "teacher_main_state_sha256": teacher.get("main_state_sha256"),
        "teacher_version": teacher.get("version"),
        "teacher_signature": teacher.get("teacher_signature"),
        "experts": list(teacher.get("experts", ())),
    }
    p = _provenance_path(ckpt)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Stage3 provenance] saved: {p}")


def assert_stage3_provenance(ckpt, model, flowx):
    p = _provenance_path(ckpt)
    if not p.exists():
        raise RuntimeError(
            "stage3_provenance.json is missing. Rerun Stage3 with this file before Stage4.3."
        )
    obj = json.loads(p.read_text(encoding="utf-8"))
    mh = _state_dict_sha256(model)
    fh = _state_dict_sha256(flowx)
    mm = obj.get("main_state_sha256") == mh
    fm = obj.get("flowx_state_sha256") == fh
    print(f"[Stage3 provenance] main match={mm} flowx match={fm}")
    if not (mm and fm):
        raise RuntimeError(
            "Stage3 lineage mismatch. Current main.pt or flowx.pt changed. Rebuild Stage3 first."
        )


def train_stage3(base, model, flowx, vocab, main_loader, cfg, extra, ckpt, rebuild_teacher=False):
    model.eval(); flowx.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in flowx.parameters():
        p.requires_grad_(False)

    teacher = precompute_stage3_teacher(
        base, model, vocab, main_loader.dataset, cfg, extra,
        _teacher_cache_path(ckpt), rebuild=rebuild_teacher,
    )
    tprobs = teacher["probs"]
    spread = teacher["effects"].amax(-1)-teacher["effects"].amin(-1)
    reliability = (spread / 0.25).clamp(0,1) * (teacher["effects"].amax(-1)>0).float()

    controller = Stage3CausalBias(cfg, extra).to(cfg.device)
    opt = torch.optim.AdamW(controller.parameters(), lr=extra.stage3_lr)
    pad = vocab.stoi[vocab.PAD]
    train_loader = DataLoader(IndexedDataset(main_loader.dataset), batch_size=cfg.batch, shuffle=True, drop_last=False)

    def run_epoch(ep, warmup):
        controller.train()
        sums = {"loss": 0.0, "gate": 0.0, "answer": 0.0, "reg": 0.0}
        nb = 0
        for img, q, a, idx in train_loader:
            img, q, a = img.to(cfg.device), q.to(cfg.device), a.to(cfg.device)
            target_gate = tprobs[idx.long()].to(cfg.device)
            gate_w = _sample_gate_weights(base, main_loader.dataset, idx, cfg, extra, cfg.device)
            gate_w = gate_w * reliability[idx.long()].to(cfg.device)
            with torch.no_grad():
                z, _ = model.build_Z(img, q)
                h = flowx_hidden(flowx, z)

            gl = controller.gate_logits(z, h)
            gate_ps = _soft_target_ce_per_sample(gl, target_gate)
            gate_loss = (gate_ps * gate_w).sum() / gate_w.sum().clamp_min(1e-8)
            answer_loss = torch.zeros((), device=cfg.device)
            reg = torch.zeros((), device=cfg.device)

            if warmup:
                loss = gate_loss
            else:
                bias_row = controller.bias_row(z, h, gl)
                ans_in, ans_out = a[:, :-1], a[:, 1:]
                cross_bias = _expand_cross_bias(bias_row, ans_in.size(1), cfg.n_heads)
                logits = model.decode_from_Z(z, ans_in, cross_bias=cross_bias)
                answer_loss = ce_loss(logits, ans_out, pad)
                reg = bias_row.pow(2).mean()
                loss = (
                    extra.stage3_gate_lambda * gate_loss
                    + extra.stage3_answer_lambda * answer_loss
                    + extra.stage3_bias_reg * reg
                )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()

            sums["loss"] += float(loss.item())
            sums["gate"] += float(gate_loss.item())
            sums["answer"] += float(answer_loss.item())
            sums["reg"] += float(reg.item())
            nb += 1

        phase = "warmup" if warmup else "joint"
        print(
            f"[Stage3][{phase}] ep{ep} loss={sums['loss']/nb:.6f} "
            f"gate={sums['gate']/nb:.6f} answer={sums['answer']/nb:.6f} reg={sums['reg']/nb:.6f} "
            f"aff={F.softplus(controller.raw_affinity_scale).item():.3f} "
            f"gateScale={F.softplus(controller.raw_gate_scale).item():.3f}"
        )

    for ep in range(1, extra.epochs_stage3_warmup + 1):
        run_epoch(ep, True)
    for ep in range(1, extra.epochs_stage3_joint + 1):
        run_epoch(ep, False)

    evaluate_stage3_gate_fit(base, model, flowx, controller, teacher, main_loader.dataset, vocab, cfg, extra)
    write_stage3_provenance(ckpt, model, flowx, teacher)
    return controller, teacher


# =====================================================================
# Stage4.3 helpers
# =====================================================================
def content_query_mask(a, vocab):
    # Shared with Stage1/2: only actual attribute values, never sentence grammar.
    return vocab.value_mask(a[:, 1:])


def contribution_expert_distribution(contribs, cfg):
    n = len(cfg.experts)
    probs = []
    for cc in contribs:
        if cc.dim() != 4 or cc.size(2) != n:
            raise ValueError(f"expected (B,Lq,{n},D), got {tuple(cc.shape)}")
        r = torch.linalg.vector_norm(cc, ord=2, dim=-1)
        p = (r + 1e-8) / (r.sum(dim=-1, keepdim=True) + n * 1e-8)
        probs.append(p)
    out = torch.stack(probs, dim=0).mean(dim=0)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def causal_contribution_kl(gate_probs, contrib_expert, query_mask):
    g = gate_probs.unsqueeze(1).expand_as(contrib_expert).clamp_min(1e-8)
    c = contrib_expert.clamp_min(1e-8)
    per_q = (g * (g.log() - c.log())).sum(dim=-1)
    m = query_mask.to(per_q.dtype)
    return (per_q * m).sum() / m.sum().clamp_min(1.0)


def weighted_contrib_kl(gate, cd, qm, weights):
    g = gate.unsqueeze(1).expand_as(cd).clamp_min(1e-8)
    c = cd.clamp_min(1e-8)
    pq = (g * (g.log() - c.log())).sum(-1)
    m = qm.to(pq.dtype)
    ps = (pq * m).sum(1) / m.sum(1).clamp_min(1.0)
    w = weights.to(ps.dtype)
    return (ps * w).sum() / w.sum().clamp_min(1e-8)


def output_invariance_loss(clean_logits, pert_logits, qm):
    p = F.softmax(clean_logits.detach(), -1).clamp_min(1e-8)
    logq = F.log_softmax(pert_logits, -1)
    pq = (p * (p.log() - logq)).sum(-1)
    m = qm.to(pq.dtype)
    return (pq * m).sum() / m.sum().clamp_min(1.0)


def direction_loss(clean_cs, pert_cs, target_idx, qm):
    b = torch.arange(target_idx.numel(), device=target_idx.device)
    m = qm.to(torch.float32)
    losses = []
    for c, p in zip(clean_cs, pert_cs):
        cv = c[b, :, target_idx, :].detach()
        pv = p[b, :, target_idx, :]
        cos = F.cosine_similarity(cv, pv, dim=-1, eps=1e-8)
        losses.append(((1.0 - cos) * m).sum() / m.sum().clamp_min(1.0))
    if not losses:
        return torch.zeros((), device=target_idx.device)
    return torch.stack(losses).mean()


def _affine(img, theta):
    x = (img + 1.0).unsqueeze(0)
    theta = theta.to(device=img.device, dtype=img.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, size=x.shape, align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (y.squeeze(0) - 1.0).clamp(-1.0, 1.0)


def _foreground_mask_one(img, threshold):
    return (img > threshold).any(dim=0)


def perturb_factor_input(img, factor, strength, sample_index, cfg):
    """Semantic-ish perturbation used ONLY on the selected expert branch."""
    if strength <= 0:
        return img.clone()

    if factor == "color":
        out = img.clone()
        fg = _foreground_mask_one(out, cfg.foreground_threshold)
        if not fg.any():
            return out
        vals = out[:, fg]
        lum = vals.mean(dim=0, keepdim=True)
        out[:, fg] = (1.0 - strength) * vals + strength * lum.expand_as(vals)
        return out.clamp(-1, 1)

    if factor == "shape":
        sign = -1.0 if sample_index % 2 else 1.0
        shear = sign * min(0.45, 0.85 * strength)
        return _affine(img, torch.tensor([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], device=img.device))

    if factor == "size":
        scale = (1.0 + 0.75 * strength) if sample_index % 2 == 0 else max(0.55, 1.0 - 0.55 * strength)
        return _affine(img, torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0]], device=img.device))

    if factor == "pattern":
        # Discrete direction intervention; not a weak crosshatch blend.
        return torch.rot90(img, k=1 if sample_index%2==0 else 3, dims=(-2,-1))

    if factor == "spacing":
        fg = _foreground_mask_one(img,cfg.foreground_threshold)
        yy,xx = torch.where(fg)
        if not yy.numel(): return img.clone()
        y0,y1,x0,x1 = int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1
        crop=img[:,y0:y1,x0:x1]
        h,w=crop.shape[-2:]
        ratio=1.0+(1 if sample_index%2==0 else -1)*max(0.25,strength)
        ys=((torch.arange(h,device=img.device)-(h-1)/2)*ratio+(h-1)/2).round().long().clamp(0,h-1)
        xs=((torch.arange(w,device=img.device)-(w-1)/2)*ratio+(w-1)/2).round().long().clamp(0,w-1)
        remap=crop[:,ys[:,None],xs[None,:]]
        out=img.clone(); old=out[:,y0:y1,x0:x1]
        # This is a branch-input stress test, not a guaranteed physical do(spacing).
        valid=fg[y0:y1,x0:x1] & (remap>cfg.foreground_threshold).any(0)
        out[:,y0:y1,x0:x1]=torch.where(valid[None],remap,old)
        return out

    raise KeyError(factor)


def choose_distractor_sets(qtypes, cfg, extra, rng):
    expert_names = list(cfg.experts)
    out = []
    for qt in qtypes:
        if qt not in expert_names:
            out.append([])
            continue
        candidates = [e for e in expert_names if e != qt]
        kmax = min(extra.stage43_max_distractors, len(candidates))
        kmin = min(extra.stage43_min_distractors, kmax)
        k = rng.randint(kmin, kmax)
        out.append(rng.sample(candidates, k=k))
    return out


@torch.no_grad()
def build_perturbed_Z(model, img, q, sms_clean, qtypes, distractor_sets, strengths, base_index, cfg):
    selected = [i for i, qt in enumerate(qtypes) if qt in cfg.experts]
    if not selected:
        return None, None, None, 0.0

    st = torch.tensor(selected, device=img.device, dtype=torch.long)
    qsub = q.index_select(0, st)
    isub = img.index_select(0, st)
    sms = [x.index_select(0, st).clone() for x in sms_clean]
    target_names = [qtypes[i] for i in selected]
    sub_dsets = [distractor_sets[i] for i in selected]

    for ei, factor in enumerate(cfg.experts):
        rows = [r for r, ds in enumerate(sub_dsets) if factor in ds]
        if not rows:
            continue
        pimgs, qs = [], []
        for r in rows:
            orig_i = selected[r]
            pimgs.append(
                perturb_factor_input(
                    isub[r], factor, float(strengths[orig_i]), base_index + orig_i, cfg
                )
            )
            qs.append(qsub[r])
        _, sm = model.experts[factor](torch.stack(pimgs), torch.stack(qs))
        rr = torch.tensor(rows, device=img.device, dtype=torch.long)
        sms[ei][rr] = sm

    zpert = torch.cat(sms, dim=1)
    target_idx = torch.tensor([list(cfg.experts).index(t) for t in target_names], device=img.device)
    zclean = torch.cat([x.index_select(0, st) for x in sms_clean], dim=1)

    # Guard: target expert block must remain byte-identical at tensor level.
    cl = cfg.compressed_len
    maxdiff = 0.0
    for r, ti in enumerate(target_idx.tolist()):
        sl = slice(ti * cl, (ti + 1) * cl)
        maxdiff = max(maxdiff, float((zpert[r:r+1, sl] - zclean[r:r+1, sl]).abs().max().item()))
    return zpert, st, target_idx, maxdiff


def set_cross_attn_trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)
    named = []
    for li, layer in enumerate(model.layers):
        for n, p in layer.cross_attn.named_parameters():
            p.requires_grad_(True)
            named.append((f"layers.{li}.cross_attn.{n}", p))
    print(f"[Stage4.3] trainable cross_attn tensors={len(named)} params={sum(p.numel() for _,p in named):,}")
    return named


def anchor_loss(trainable_named, initial_params):
    if not trainable_named:
        return torch.zeros(())
    vals = [(p - initial_params[n]).pow(2).mean() for n, p in trainable_named]
    return torch.stack(vals).mean()


@torch.no_grad()
def stage43_alignment(base, model, flowx, controller, vocab, dataset, cfg, limit=0):
    model.eval(); flowx.eval(); controller.eval()
    loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=False, drop_last=False)
    sums = {"kl": 0.0, "agree": 0, "n": 0, "routing": 0, "routing_n": 0}
    seen = 0
    expert_names = list(cfg.experts)

    for img, q, a in loader:
        if limit > 0 and seen >= limit:
            break
        if limit > 0 and seen + img.size(0) > limit:
            keep = limit - seen
            img, q, a = img[:keep], q[:keep], a[:keep]
        img, q, a = img.to(cfg.device), q.to(cfg.device), a.to(cfg.device)
        z, _ = model.build_Z(img, q)
        gp = controller.gate_probs(z, flowx_hidden(flowx, z))
        _logits, _attn, contribs = model.decode_from_Z(
            z, a[:, :-1], return_cross_contrib=True
        )
        cd = contribution_expert_distribution(contribs, cfg)
        qm = content_query_mask(a, vocab)
        gq = gp.unsqueeze(1).expand_as(cd).clamp_min(1e-8)
        cq = cd.clamp_min(1e-8)
        per = (gq * (gq.log() - cq.log())).sum(-1)
        m = qm.to(per.dtype)
        ps = (per * m).sum(1) / m.sum(1).clamp_min(1.0)
        sums["kl"] += ps.sum().item()

        # Mean contribution over content query positions for argmax comparison.
        cm = (cd * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
        sums["agree"] += (cm.argmax(-1) == gp.argmax(-1)).sum().item()
        sums["n"] += img.size(0)

        for b in range(img.size(0)):
            qt = question_type_from_q(base, q[b], vocab, cfg)
            if qt in expert_names:
                sums["routing_n"] += 1
                pred = expert_names[int(cm[b].argmax().item())]
                sums["routing"] += int(pred == qt)

        seen += img.size(0)

    n = max(1, sums["n"])
    return {
        "contribution_kl": sums["kl"] / n,
        "contribution_argmax": sums["agree"] / n,
        "expected_factor_routing": sums["routing"] / max(1, sums["routing_n"]),
        "n": sums["n"],
    }


def train_stage43(base, model, flowx, controller, vocab, main_loader, cfg, extra):
    model.eval()
    use_teacher = flowx is not None and controller is not None
    for module in (flowx,controller):
        if module is not None:
            module.eval()
            for p in module.parameters(): p.requires_grad_(False)

    trainable = set_cross_attn_trainable(model)
    initial = {n: p.detach().clone() for n, p in trainable}
    opt = torch.optim.AdamW([p for _, p in trainable], lr=extra.stage43_lr, weight_decay=0.0)
    pad = vocab.stoi[vocab.PAD]
    rng = random.Random(extra.stage43_seed)

    print("[Stage4.3] DIRECT Stage1 init; NO Stage3 bias in student forward")
    print("[Stage4.3] single-factor Q only: target expert CLEAN, random 1-2 of 4 distractors perturbed")
    print(f"[Stage4.3] perturb train strength={extra.stage43_perturb_min:.2f}-{extra.stage43_perturb_max:.2f}")
    print(f"[Stage4.3] main-question clean contribution scale={extra.stage43_main_contrib_scale:.2f}")
    if not use_teacher: print("[Stage4.3] FlowX/teacher disabled; reported contribution KL/agreement zeros below are placeholders, not evaluation scores")

    before = stage43_alignment(
        base, model, flowx, controller, vocab, main_loader.dataset, cfg,
        limit=min(300, len(main_loader.dataset))
    ) if use_teacher else {"contribution_kl":0.,"contribution_argmax":0.,"expected_factor_routing":0.,"disabled":True}
    print(
        f"[Stage4.3] pre contribKL={before['contribution_kl']:.6f} "
        f"argmax={before['contribution_argmax']:.2%} routing={before['expected_factor_routing']:.2%}"
    )

    global_index = 0
    for ep in range(1, extra.epochs_stage43 + 1):
        ramp = min(1.0, ep / max(1, extra.stage43_warmup_epochs))
        hi = extra.stage43_perturb_min + ramp * (extra.stage43_perturb_max - extra.stage43_perturb_min)
        sums = {k: 0.0 for k in ["loss","answer","cleanC","pertCE","inv","pertC","dir","anchor"]}
        nb = 0
        guard = 0.0

        for img, q, a in main_loader:
            img, q, a = img.to(cfg.device), q.to(cfg.device), a.to(cfg.device)
            qtypes = [question_type_from_q(base, q[i], vocab, cfg) for i in range(q.size(0))]
            strengths = [rng.uniform(extra.stage43_perturb_min, hi) for _ in range(img.size(0))]
            dsets = choose_distractor_sets(qtypes, cfg, extra, rng)

            with torch.no_grad():
                z, sms = model.build_Z(img, q)
                gate = controller.gate_probs(z, flowx_hidden(flowx, z)).detach() if use_teacher else None

            logits, _aw, cs = model.decode_from_Z(
                z, a[:, :-1], cross_bias=None, return_cross_contrib=True
            )
            qm = content_query_mask(a, vocab)
            answer = ce_loss(logits, a[:, 1:], pad)
            cd = contribution_expert_distribution(cs, cfg)

            weights = torch.tensor(
                [1.0 if qt in cfg.experts else extra.stage43_main_contrib_scale for qt in qtypes],
                device=cfg.device,
            )
            cleanC = weighted_contrib_kl(gate, cd, qm, weights) if use_teacher else answer*0.0

            zp, sel, target_idx, md = build_perturbed_Z(
                model, img, q, sms, qtypes, dsets, strengths, global_index, cfg
            )
            guard = max(guard, md)
            if md != 0.0: raise RuntimeError("Target expert changed during distractor intervention")

            if zp is not None:
                ap = a.index_select(0, sel)
                qmp = content_query_mask(ap, vocab)
                gp = gate.index_select(0, sel) if use_teacher else None
                lclean = logits.index_select(0, sel)
                cclean = [x.index_select(0, sel) for x in cs]

                lp, _apw, cp = model.decode_from_Z(
                    zp, ap[:, :-1], cross_bias=None, return_cross_contrib=True
                )
                pertCE = ce_loss(lp, ap[:, 1:], pad)
                inv = output_invariance_loss(lclean, lp, qmp)
                pcd = contribution_expert_distribution(cp, cfg)
                pertC = causal_contribution_kl(gp, pcd, qmp) if use_teacher else answer*0.0
                direc = direction_loss(cclean, cp, target_idx, qmp)
            else:
                zero = answer * 0.0
                pertCE = inv = pertC = direc = zero

            anch = anchor_loss(trainable, initial).to(cfg.device)
            loss = (
                extra.stage43_answer_lambda * answer
                + ramp * extra.stage43_clean_contrib_lambda * cleanC
                + ramp * extra.stage43_pert_answer_lambda * pertCE
                + ramp * extra.stage43_output_invariance_lambda * inv
                + ramp * extra.stage43_pert_contrib_lambda * pertC
                + ramp * extra.stage43_direction_lambda * direc
                + extra.stage43_anchor_lambda * anch
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], extra.stage43_grad_clip)
            opt.step()

            for k, v in [
                ("loss", loss), ("answer", answer), ("cleanC", cleanC),
                ("pertCE", pertCE), ("inv", inv), ("pertC", pertC),
                ("dir", direc), ("anchor", anch),
            ]:
                sums[k] += float(v.item())
            nb += 1
            global_index += img.size(0)

        print(
            f"[Stage4.3] ep{ep:02d} loss={sums['loss']/nb:.6f} "
            f"answer={sums['answer']/nb:.6f} cleanC={sums['cleanC']/nb:.6f} "
            f"pertCE={sums['pertCE']/nb:.6f} inv={sums['inv']/nb:.6f} "
            f"pertC={sums['pertC']/nb:.6f} dir={sums['dir']/nb:.6f} "
            f"anchor={sums['anchor']/nb:.8f} ramp={ramp:.2f} targetMaxDiff={guard:.2e}"
        )

    after = stage43_alignment(
        base, model, flowx, controller, vocab, main_loader.dataset, cfg,
        limit=min(300, len(main_loader.dataset))
    ) if use_teacher else dict(before)
    print(
        f"[Stage4.3] post contribKL={after['contribution_kl']:.6f} "
        f"argmax={after['contribution_argmax']:.2%} routing={after['expected_factor_routing']:.2%}"
    )
    return model, {"before": before, "after": after,"uses_flowx_teacher":use_teacher}


# =====================================================================
# Unseen summary evaluator: Stage1 vs Stage4.3, no Stage3 bias
# =====================================================================
@torch.no_grad()
def generate(model, vocab, img, q, cfg):
    model.eval()
    img = img.unsqueeze(0).to(cfg.device)
    q = q.unsqueeze(0).to(cfg.device)
    z, _ = model.build_Z(img, q)
    ids = [vocab.stoi[vocab.BOS]]
    for _ in range(cfg.max_answer_len - 1):
        ans_in = torch.tensor([ids], device=cfg.device)
        logits = model.decode_from_Z(z, ans_in)
        nxt = int(logits[:, -1].argmax(-1).item())
        ids.append(nxt)
        if nxt == vocab.stoi[vocab.EOS]:
            break
    return vocab.decode(ids)


@torch.no_grad()
def eval_unseen_summary(base, model, vocab, dataset, cfg, label="model", limit=0, verbose=False):
    factor_order = base.FACTOR_ORDER
    n = len(dataset) if limit <= 0 else min(limit, len(dataset))
    correct = {f: 0 for f in factor_order}
    joint = 0
    exact = 0
    for i in range(n):
        img, q, _a = dataset[i]
        _path, _prompt, gt = dataset.items[i]
        pred = generate(model, vocab, img, q, cfg)
        gt_map = base.parse_structured_answer(gt)
        pr_map = base.parse_structured_answer(pred)
        oks = []
        factor_oks = base.factor_correctness(gt, pred)
        for f in factor_order:
            ok = factor_oks[f]
            correct[f] += int(ok)
            oks.append(ok)
        joint += int(all(oks))
        exact += int(pred == gt)
        if verbose:
            print(f"{dataset.items[i][0].stem:<14} GT={gt} | pred={pred} | {'o' if all(oks) else 'x'}")

    out = {f: correct[f] / max(1, n) for f in factor_order}
    out["joint"] = joint / max(1, n)
    out["exact"] = exact / max(1, n)
    print(f"[{label}] n={n}")
    for f in factor_order:
        print(f"  {f:<8}: {correct[f]}/{n} = {out[f]:.2%}")
    print(f"  {'joint':<8}: {joint}/{n} = {out['joint']:.2%}")
    print(f"  {'exact':<8}: {exact}/{n} = {out['exact']:.2%}")
    return out


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True,
                    help="Path to expert_flowx_5experts_v9_compact.py")
    ap.add_argument("--invariance-only", action="store_true", help="Stage43 control: no FlowX, no CausalGate, no contribution KL")
    ap.add_argument("--root", type=str,
                    default="ここにこのファイルがあるディレクトリのパス")
    ap.add_argument("--ckpt", type=str,
                    default="ここに前回までのチェックポイントのパス")
    ap.add_argument("--stage", type=str, choices=["3", "43"], default=None)
    ap.add_argument("--rebuild-causal-cache", action="store_true")

    ap.add_argument("--epochs-stage3-warmup", type=int, default=15)
    ap.add_argument("--epochs-stage3-joint", type=int, default=35)
    ap.add_argument("--epochs-stage43", type=int, default=30)
    ap.add_argument("--stage43-seed", type=int, default=4305)

    ap.add_argument("--eval-unseen", action="store_true")
    ap.add_argument("--eval-heldout", action="store_true")
    ap.add_argument("--compare-stage1-stage43", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--verbose-eval", action="store_true")
    args = ap.parse_args()

    base = load_base_module(args.base_model)
    cfg = base.CFG_
    cfg.root = args.root
    base.load_manifest_config(args.ckpt,cfg)
    base.seed_all(cfg.seed)
    extra = ExtraCFG(
        epochs_stage3_warmup=args.epochs_stage3_warmup,
        epochs_stage3_joint=args.epochs_stage3_joint,
        epochs_stage43=args.epochs_stage43,
        stage43_seed=args.stage43_seed,
    )

    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    vocab, expert_loaders, main_loader, route_loader, unseen_seen, unseen_held = base.build_loaders(cfg)

    base.verify_run_manifest(args.ckpt,cfg,vocab)
    main_path = Path(args.ckpt) / "main.pt"
    flowx_path = Path(args.ckpt) / "flowx.pt"
    stage3_path = Path(args.ckpt) / "stage3.pt"
    stage43_path = Path(args.ckpt) / ("main_stage43_invariance.pt" if args.invariance_only else "main_stage43.pt")

    if not main_path.exists():
        raise FileNotFoundError(main_path)
    needs_flowx = args.stage=="3" or (args.stage=="43" and not args.invariance_only)
    if needs_flowx and not flowx_path.exists(): raise FileNotFoundError(flowx_path)

    # Load current Stage1 checkpoint into contribution-aware architecture.
    model = CausalMainFlow(base, cfg, len(vocab)).to(cfg.device)
    missing, unexpected = model.load_state_dict(torch.load(main_path, map_location=cfg.device), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"main.pt compatibility failure: missing={missing} unexpected={unexpected}")

    flowx = None
    if needs_flowx:
        flowx = base.FlowXGate(cfg).to(cfg.device)
        flowx.load_state_dict(torch.load(flowx_path,map_location=cfg.device,weights_only=True))

    print("=" * 100)
    print("5-EXPERT Stage3 -> CausalGate -> Stage4.3")
    print("=" * 100)
    print(f"root       : {cfg.root}")
    print(f"ckpt       : {args.ckpt}")
    print(f"base model : {args.base_model}")
    print(f"main hash  : {_state_dict_sha256(model)}")
    print(f"flowx hash : {_state_dict_sha256(flowx) if flowx is not None else 'not loaded'}")
    print(f"experts    : {cfg.experts}")

    if args.stage == "3":
        controller, teacher = train_stage3(
            base, model, flowx, vocab, main_loader, cfg, extra,
            args.ckpt, rebuild_teacher=args.rebuild_causal_cache,
        )
        torch.save(controller.state_dict(), stage3_path)
        print(f"[saved] {stage3_path}")
        return

    if args.stage == "43":
        controller = None
        if not args.invariance_only:
            if not stage3_path.exists(): raise FileNotFoundError(stage3_path)
            assert_stage3_provenance(args.ckpt,model,flowx)
            controller = Stage3CausalBias(cfg,extra).to(cfg.device)
            controller.load_state_dict(torch.load(stage3_path,map_location=cfg.device,weights_only=True))

        # IMPORTANT: fresh Stage1 init, never continue from an old Stage4.x model.
        stage43_model = CausalMainFlow(base, cfg, len(vocab)).to(cfg.device)
        stage43_model.load_state_dict(torch.load(main_path, map_location=cfg.device))

        stage43_model, meta = train_stage43(
            base, stage43_model, flowx, controller, vocab, main_loader, cfg, extra
        )
        torch.save(stage43_model.state_dict(), stage43_path)
        meta.update({
            "version": 1,
            "main_state_sha256": _state_dict_sha256(model),
            "flowx_state_sha256": _state_dict_sha256(flowx) if flowx is not None else None,
            "stage3_state_sha256": _state_dict_sha256(controller) if controller is not None else None,
            "stage43_state_sha256": _state_dict_sha256(stage43_model),
            "experts": list(cfg.experts),
            "seed": extra.stage43_seed,
        })
        torch.save(meta, Path(args.ckpt) / ("stage43_invariance_meta.pt" if args.invariance_only else "stage43_meta.pt"))
        print(f"[saved] {stage43_path}")
        print(f"[saved] metadata for {stage43_path.name}")
        return

    # Evaluation-only path.
    if args.eval_unseen or args.eval_heldout or args.compare_stage1_stage43:
        if args.compare_stage1_stage43:
            if not stage43_path.exists():
                raise FileNotFoundError(stage43_path)
            m43 = CausalMainFlow(base, cfg, len(vocab)).to(cfg.device)
            m43.load_state_dict(torch.load(stage43_path, map_location=cfg.device))

            splits = []
            if args.eval_unseen or (not args.eval_unseen and not args.eval_heldout):
                splits.append(("union_seen_components", unseen_seen))
            if args.eval_heldout:
                splits.append(("union_heldout_cs", unseen_held))

            for split_name, ds in splits:
                print("\n" + "-" * 100)
                print(f"[{split_name}] Stage1 vs Stage4.3; NO Stage3 bias")
                print("-" * 100)
                b = eval_unseen_summary(base, model, vocab, ds, cfg, "Stage1", args.eval_limit, args.verbose_eval)
                s = eval_unseen_summary(base, m43, vocab, ds, cfg, "Stage4.3", args.eval_limit, args.verbose_eval)
                print("  delta Stage4.3-Stage1")
                for k in list(base.FACTOR_ORDER) + ["joint", "exact"]:
                    print(f"    {k:<8}: {s[k]-b[k]:+.2%}")
            return

        if args.eval_unseen:
            eval_unseen_summary(base, model, vocab, unseen_seen, cfg, "Stage1", args.eval_limit, args.verbose_eval)
        if args.eval_heldout:
            eval_unseen_summary(base, model, vocab, unseen_held, cfg, "Stage1-heldout", args.eval_limit, args.verbose_eval)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
