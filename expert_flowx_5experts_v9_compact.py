# -*- coding: utf-8 -*-
"""Five-expert v9: compact Japanese answers, minimal expert / joint-main data.
The encoder, decoder and FlowX architecture are preserved. Tokenization,
expert class targets, content masks and evaluation use factor_text_v9.
Use a new checkpoint directory and retrain Stage0 onward.
"""

import os
import copy
import random
import hashlib
import numpy as np
import math
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from factor_text_v9 import (TEXT_FORMAT_VERSION, FACTOR_ORDER, FACTOR_VALUES, VALUE_WORDS,
                            NULL_VALUE, tokenize, parse_answer, parse_single_answer,
                            factor_correctness)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class CFG:
    root: str = "ルートディレクトリ"

    dir_shape: str = "shape"
    dir_color: str = "color"
    dir_size: str = "size"
    dir_pattern: str = "pattern"
    dir_spacing: str = "spacing"

    dir_main: str = "main_cont"
    dir_main_text: str = "main_text_cont"

    dir_shape_text: str = "shape_text"
    dir_color_text: str = "color_text"
    dir_size_text: str = "size_text"
    dir_pattern_text: str = "pattern_text"
    dir_spacing_text: str = "spacing_text"

    dir_unseen_seen: str = "unseen_union_seen_components"
    dir_unseen_seen_text: str = "unseen_union_seen_components_text"
    dir_unseen_heldout: str = "unseen_union_heldout_cs"
    dir_unseen_heldout_text: str = "unseen_union_heldout_cs_text"

    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 2
    dec_layers: int = 4
    compressed_len: int = 8
    max_text_len: int = 40
    max_answer_len: int = 24
    img_size: int = 64  # CNN input AFTER factor canonicalization
    input_size: int = 256  # preserve fine stripes BEFORE bbox/profile extraction
    expert_min_steps: int = 200
    expert_check_every: int = 25
    expert_stable_checks: int = 3
    expert_max_ce: float = 0.08
    seed: int = 1
    format_version: str = TEXT_FORMAT_VERSION

    batch: int = 16
    lr: float = 3e-4
    epochs_expert: int = 500
    epochs_stage1: int = 20
    epochs_stage2: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    freeze_experts_stage1: bool = True
    flowx_lambda: float = 0.0

    color_kw: str = "色"
    shape_kw: str = "形状"
    size_kw: str = "大きさ"
    pattern_kw: str = "模様"
    spacing_kw: str = "間隔"
    main_kw: str = "これは何ですか"

    experts: tuple = ("shape", "color", "size", "pattern", "spacing")
    expert_transforms: dict = field(default_factory=lambda: {
        "shape": "shape_normalized",
        "color": "color_patch",
        "size": "size_radius",
        "pattern": "pattern_profile",
        "spacing": "pattern_profile",
    })

    foreground_threshold: float = -0.95
    shape_mask_value: float = 220.0 / 127.5 - 1.0
    shape_normalized_extent: float = 0.60


CFG_ = CFG()


# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------
class WordVocab:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self):
        self.stoi = {t: i for i, t in enumerate([self.PAD, self.BOS, self.EOS, self.UNK])}
        self.itos = list(self.stoi)

    def build(self, texts):
        for text in texts:
            if "<" in text or ">" in text:
                raise ValueError("v9 requires compact Japanese data without factor tags. Regenerate the dataset.")
            for word in tokenize(text):
                if word not in self.stoi:
                    self.stoi[word] = len(self.itos)
                    self.itos.append(word)
        return self

    def encode(self, text, max_len, add_special=True):
        ids = [self.stoi.get(w, self.stoi[self.UNK]) for w in tokenize(text)]
        if add_special:
            ids = [self.stoi[self.BOS]] + ids + [self.stoi[self.EOS]]
        if len(ids) > max_len:
            raise ValueError(f"Text needs {len(ids)} tokens, max_len={max_len}: {text!r}")
        ids += [self.stoi[self.PAD]] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok == self.EOS:
                break
            if tok not in (self.PAD, self.BOS):
                out.append(tok)
        return "".join(out)

    def value_mask(self, target):
        mask = torch.zeros_like(target, dtype=torch.bool)
        for value in VALUE_WORDS:
            if value in self.stoi:
                mask |= target == self.stoi[value]
        return mask

    def __len__(self):
        return len(self.itos)


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _read_sample(txt_dir: Path, stem: str):
    p = txt_dir / f"{stem}.txt"
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    prompt = lines[0].strip()
    label = lines[1].strip() if len(lines) > 1 else stem.split("_")[0]
    return prompt, label


def image_to_tensor(image, size):
    image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1.0


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def content_mask(target, vocab):
    # Attribute values only: do not distill grammar, punctuation or EOS.
    return vocab.value_mask(target)


def expert_class_targets(a, vocab):
    mask = vocab.value_mask(a)
    if not bool((mask.sum(-1) == 1).all()):
        raise ValueError("Expert answers must contain exactly one attribute value.")
    return a.gather(1, mask.long().argmax(-1, keepdim=True)).squeeze(1)


class PairDataset(Dataset):
    def __init__(self, img_dir, txt_dir, vocab, cfg: CFG, fixed_prompt=None):
        self.img_dir, self.txt_dir = Path(img_dir), Path(txt_dir)
        self.vocab, self.cfg = vocab, cfg
        self.fixed_prompt = fixed_prompt
        self.tf = lambda image: image_to_tensor(image, cfg.input_size)
        self.items = []
        for f in sorted(self.img_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT and (self.txt_dir / f"{f.stem}.txt").exists():
                prompt, label = _read_sample(self.txt_dir, f.stem)
                self.items.append((f, prompt, label))
        if not self.items:
            raise RuntimeError(
                f"ペアが0件です: img_dir={self.img_dir} txt_dir={self.txt_dir}\n"
                "フォルダ名・ファイル名対応を確認してください。"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, prompt, label = self.items[i]
        img = self.tf(Image.open(path).convert("RGB"))
        if self.fixed_prompt is not None:
            prompt = self.fixed_prompt
        q = self.vocab.encode(prompt, self.cfg.max_text_len)
        a = self.vocab.encode(label, self.cfg.max_answer_len)
        return img, q, a


class FactorRouteDataset(Dataset):
    """main_cont から factor-specific question だけを抜き出し、ルーティング教師を返す。"""
    def __init__(self, img_dir, txt_dir, vocab, cfg: CFG):
        self.base = PairDataset(img_dir, txt_dir, vocab, cfg)
        self.items = []  # (base_idx, expert_name)
        for idx, (_, prompt, _) in enumerate(self.base.items):
            qtype = question_type(prompt, cfg)
            if qtype in cfg.experts:
                self.items.append((idx, qtype))
        if not self.items:
            raise RuntimeError("FactorRouteDataset が空です。main_cont に factor 質問があるか確認してください。")
        self.name_to_idx = {name: i for i, name in enumerate(cfg.experts)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        base_idx, expert = self.items[i]
        img, q, a = self.base[base_idx]
        target = torch.tensor(self.name_to_idx[expert], dtype=torch.long)
        return img, q, a, target


# ------------------------------------------------------------------
# Input transforms for physical factor isolation
# ------------------------------------------------------------------
def _foreground_mask(img: torch.Tensor, threshold: float = -0.95) -> torch.Tensor:
    return (img > threshold).any(dim=1, keepdim=True)


def _crop_bbox(mask2d: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = torch.where(mask2d)
    if ys.numel() == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return y0, y1, x0, x1


def _render_shape_normalized(img: torch.Tensor, cfg: CFG) -> torch.Tensor:
    fg = _foreground_mask(img, cfg.foreground_threshold)
    B, _, H, W = img.shape
    out = torch.full_like(img, -1.0)
    target_extent = max(4, int(round(min(H, W) * cfg.shape_normalized_extent)))
    fg_value = float(cfg.shape_mask_value)

    for b in range(B):
        box = _crop_bbox(fg[b, 0])
        if box is None or bool(fg[b, 0].all()):
            continue
        y0, y1, x0, x1 = box
        crop = fg[b:b+1, :, y0:y1+1, x0:x1+1].float()
        h0, w0 = crop.shape[-2:]
        scale = target_extent / max(h0, w0)
        nh = max(1, int(round(h0 * scale)))
        nw = max(1, int(round(w0 * scale)))
        rs = F.interpolate(crop, size=(nh, nw), mode="bilinear", align_corners=False)
        yy = (H - nh) // 2
        xx = (W - nw) // 2
        patch = (rs[0, 0] > 0.5)
        for c in range(3):
            out[b, c, yy:yy+nh, xx:xx+nw][patch] = fg_value
    return out


def _render_color_patch(img: torch.Tensor, cfg: CFG) -> torch.Tensor:
    # Bright-band statistic, rather than averaging bright and dark bands.
    # This is a renderer-specific prior; it is NOT a general color constancy model.
    fg = _foreground_mask(img, cfg.foreground_threshold)
    out = torch.full_like(img, -1.0)
    for b in range(img.size(0)):
        vals = img[b, :, fg[b, 0]]
        if vals.numel():
            rgb = torch.quantile(vals, 0.85, dim=1)
            out[b] = rgb[:, None, None]
    return out

def _render_size_radius(img: torch.Tensor, cfg: CFG) -> torch.Tensor:
    fg = _foreground_mask(img, cfg.foreground_threshold)
    B, _, H, W = img.shape
    yy = torch.linspace(-1, 1, H, device=img.device).view(1, H, 1)
    xx = torch.linspace(-1, 1, W, device=img.device).view(1, 1, W)
    rr = torch.sqrt(xx**2 + yy**2).expand(B, H, W)
    out = torch.full_like(img, -1.0)
    fg_value = float(cfg.shape_mask_value)

    for b in range(B):
        if bool(fg[b, 0].all()): continue  # Family B has no bounded object
        r = rr[b][fg[b, 0]]
        if r.numel() == 0:
            continue
        rad = float(r.max())
        disk = (rr[b] <= rad)
        for c in range(3):
            out[b, c][disk] = fg_value
    return out


def _to_luma(x: torch.Tensor) -> torch.Tensor:
    # x normalized to [-1,1] -> weighted luma also in about [-1,1]
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _render_pattern_profile(img: torch.Tensor, cfg: CFG) -> torch.Tensor:
    """Mask-weighted profiles in object-relative coordinates.

    Black pixels OUTSIDE the shape are excluded. Interior erosion avoids the
    anti-aliased boundary. The two axes share one contrast scale, preserving
    orientation evidence. Both experts still receive spacing AND orientation;
    independent networks alone do not prove strict factor independence.
    """
    fg = _foreground_mask(img, cfg.foreground_threshold)
    B, _, H, W = img.shape
    out = torch.full_like(img, -1.0)
    for b in range(B):
        box = _crop_bbox(fg[b, 0])
        if box is None: continue
        y0, y1, x0, x1 = box
        # Erode against the full image before cropping; a full canvas stays full.
        mask = fg[b:b+1].float()
        mask = 1.0 - F.max_pool2d(1.0 - mask, 3, 1, 1)
        m = mask[:, :, y0:y1+1, x0:x1+1]
        lum = _to_luma(img[b:b+1, :, y0:y1+1, x0:x1+1])
        def profile(axis, length):
            den = m.sum(dim=axis).flatten()
            num = (lum*m).sum(dim=axis).flatten()
            good = den >= 1
            ids = torch.where(good)[0]
            if not ids.numel(): return torch.zeros(length, device=img.device, dtype=img.dtype)
            v = num / den.clamp_min(1)
            # Fill unsupported edge coordinates from the nearest valid coordinate.
            nearest = (torch.arange(v.numel(), device=img.device)[:,None]-ids[None,:]).abs().argmin(1)
            v = torch.where(good, v, v[ids[nearest]])
            v = F.interpolate(v[None,None], size=length, mode="linear", align_corners=False).flatten()
            return v - v.mean()
        xp, yp = profile(2, W), profile(3, H)
        contrast = torch.maximum(xp.abs().max(), yp.abs().max())
        if float(contrast) < 0.02: continue  # no visible pattern
        xp = (xp / contrast).clamp(-1,1); yp = (yp / contrast).clamp(-1,1)
        xmap = xp[None,:].expand(H,W); ymap = yp[:,None].expand(H,W)
        out[b] = torch.stack([xmap,ymap,0.5*(xmap+ymap)])
    return out

def apply_input_transform(img: torch.Tensor, mode: str, cfg: CFG) -> torch.Tensor:
    if mode == "shape_normalized":
        return _render_shape_normalized(img, cfg)
    if mode == "color_patch":
        return _render_color_patch(img, cfg)
    if mode == "size_radius":
        return _render_size_radius(img, cfg)
    if mode == "pattern_profile":
        return _render_pattern_profile(img, cfg)
    return img


# ------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, d_model, img_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Conv2d(128, d_model, 4, 2, 1), nn.GELU(),
        )
        self.img_size = img_size
        n = (img_size // 16) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, d_model) * 0.02)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False, antialias=True)
        h = self.net(x)
        h = h.flatten(2).transpose(1, 2)
        return h + self.pos


class Expert(nn.Module):
    def __init__(self, cfg: CFG, vocab_size, mode="none"):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        self.img_enc = ImageEncoder(cfg.d_model, cfg.img_size)
        self.txt_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.txt_pos = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_model * 4, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.enc_layers, enable_nested_tensor=False)
        self.pool = nn.AdaptiveAvgPool1d(cfg.compressed_len)
        self.cls_head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, img, q):
        img = apply_input_transform(img, self.mode, self.cfg)
        image_tokens = self.img_enc(img)
        h = torch.cat([image_tokens, self.txt_emb(q) + self.txt_pos[:, :q.size(1)]], dim=1)
        key_padding = torch.cat([torch.zeros(image_tokens.shape[:2], device=q.device, dtype=torch.bool), q.eq(0)], 1)
        h = self.enc(h, src_key_padding_mask=key_padding)
        # Compress image positions after question-conditioned attention. PAD is
        # neither attended to nor included in the representation average.
        sm = self.pool(h[:, :image_tokens.size(1)].transpose(1, 2)).transpose(1, 2)
        return h, sm

    def pretrain_logits(self, img, q):
        _, sm = self.forward(img, q)
        return self.cls_head(sm.mean(dim=1))


class MainDecoderLayer(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4), nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model)
        )
        self.n1, self.n2, self.n3 = (nn.LayerNorm(cfg.d_model) for _ in range(3))

    def forward(self, y, z, causal_mask, cross_bias=None):
        h, _ = self.self_attn(y, y, y, attn_mask=causal_mask, need_weights=False)
        y = self.n1(y + h)
        h, _ = self.cross_attn(y, z, z, attn_mask=cross_bias, need_weights=False)
        y = self.n2(y + h)
        y = self.n3(y + self.ff(y))
        return y


class MainFlow(nn.Module):
    def __init__(self, cfg: CFG, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleDict({
            name: Expert(cfg, vocab_size, mode=cfg.expert_transforms.get(name, "none"))
            for name in cfg.experts
        })
        self.ans_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.ans_pos = nn.Parameter(torch.randn(1, cfg.max_answer_len, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList([MainDecoderLayer(cfg) for _ in range(cfg.dec_layers)])
        self.out = nn.Linear(cfg.d_model, vocab_size)

    def build_Z(self, img, q):
        sms = []
        for name in self.cfg.experts:
            _, sm = self.experts[name](img, q)
            sms.append(sm)
        z = torch.cat(sms, dim=1)
        return z, sms

    def decode_from_Z(self, z, ans_in, cross_bias=None):
        L = ans_in.size(1)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=ans_in.device), 1)
        y = self.ans_emb(ans_in) + self.ans_pos[:, :L]
        for lyr in self.layers:
            y = lyr(y, z, causal, cross_bias=cross_bias)
        return self.out(y)

    def forward(self, img, q, ans_in, cross_bias=None):
        z, _ = self.build_Z(img, q)
        return self.decode_from_Z(z, ans_in, cross_bias=cross_bias)


class FlowXGate(nn.Module):
    """FlowX-style delta-attention encoder + expert-routing head."""
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.w_o = nn.Linear(len(cfg.experts) * d, d)
        self.route_head = nn.Linear(d, len(cfg.experts))

    def _attn_weights(self, z, drop_idx=None):
        q, k, v = self.q_proj(z), self.k_proj(z), self.v_proj(z)
        scores = q @ k.transpose(1, 2) / math.sqrt(z.size(-1))
        if drop_idx is not None:
            scores[:, :, drop_idx] = float("-inf")
        return F.softmax(scores, dim=-1), v

    def encode(self, z):
        # True signed change in THIS auxiliary attention's output; not a causal
        # effect on MainDecoder. Q/K/V are projected once and reused.
        n, cl = len(self.cfg.experts), self.cfg.compressed_len
        if z.size(1) != n * cl: raise ValueError("FlowX requires all expert blocks")
        q, k, v = self.q_proj(z), self.k_proj(z), self.v_proj(z)
        scores = q @ k.transpose(1,2) / math.sqrt(z.size(-1))
        plus = scores.softmax(-1)
        heads = []
        for i in range(n):
            mask = torch.zeros(z.size(1), dtype=torch.bool, device=z.device)
            mask[i*cl:(i+1)*cl] = True
            minus = scores.masked_fill(mask[None,None], float("-inf")).softmax(-1)
            heads.append(((plus-minus) @ v).mean(dim=1))
        return self.w_o(torch.cat(heads, dim=-1))

    def forward(self, z):
        h = self.encode(z)
        logits = self.route_head(h)
        return logits, h


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def ce_loss(logits, target, pad_id):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_id)


def question_type(prompt: str, cfg: CFG) -> str:
    has_c = cfg.color_kw in prompt
    has_s = cfg.shape_kw in prompt
    has_z = cfg.size_kw in prompt
    has_p = cfg.pattern_kw in prompt
    has_d = cfg.spacing_kw in prompt

    flags = {"color": has_c, "shape": has_s, "size": has_z, "pattern": has_p, "spacing": has_d}
    active = [k for k, v in flags.items() if v]
    if len(active) == 1:
        return active[0]
    if cfg.main_kw in prompt or "何ですか" in prompt:
        return "main"
    return "mixed"


def parse_structured_answer(text: str) -> Dict[str, str]:
    # Retain the callable name for extension compatibility; no tag parsing.
    return parse_answer(text)


def _remove_expert_tokens(z: torch.Tensor, expert_idx: int, cl: int) -> torch.Tensor:
    s = expert_idx * cl
    e = (expert_idx + 1) * cl
    return torch.cat([z[:, :s, :], z[:, e:, :]], dim=1)


@torch.no_grad()
def target_content_logprob_from_Z(model: MainFlow, z, a, vocab):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    a = a.to(z.device)
    ans_in = a[:, :-1]
    target = a[:, 1:]
    logits = model.decode_from_Z(z, ans_in)
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    pad_id = vocab.stoi[vocab.PAD]
    eos_id = vocab.stoi[vocab.EOS]
    mask = content_mask(target, vocab)
    if not mask.any():
        mask = target != pad_id
    return tok_lp[mask].mean().item()


def _dominant_from_effects(effects: dict, ratio: float = 1.5, eps: float = 1e-4) -> str:
    pos = {k: max(0.0, float(v)) for k, v in effects.items()}
    order = sorted(pos.items(), key=lambda kv: kv[1], reverse=True)
    if not order or order[0][1] < eps:
        return "none"
    if len(order) == 1 or order[0][1] > order[1][1] * ratio:
        return order[0][0]
    return "mix"


# ------------------------------------------------------------------
# Train loops
# ------------------------------------------------------------------
def make_experts(cfg: CFG, vocab_size: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        name: Expert(cfg, vocab_size, mode=cfg.expert_transforms.get(name, "none"))
        for name in cfg.experts
    })


def _augment_expert(img, name):
    out = img.clone()
    if name in ("pattern", "spacing"):
        for b in range(out.size(0)):
            # Whole-canvas phase shift: valid for expert pretraining canvases.
            out[b] = torch.roll(out[b], (random.randrange(out.size(-2)), random.randrange(out.size(-1))), (-2,-1))
    # Brightness never changes the intended discrete label in this synthetic task.
    gain = torch.empty(out.size(0),1,1,1,device=out.device).uniform_(0.85,1.08)
    return ((out+1)*gain-1).clamp(-1,1)


@torch.no_grad()
def expert_accuracy(expert, dataset, vocab, cfg, prompt=None, challenge=False):
    expert.eval(); correct = total = 0; loss_sum = 0.0
    for img,q,a in DataLoader(dataset,batch_size=cfg.batch,shuffle=False):
        img,q,a = img.to(cfg.device),q.to(cfg.device),a.to(cfg.device)
        if prompt is not None: q = vocab.encode(prompt,cfg.max_text_len).to(cfg.device)[None].expand(img.size(0),-1)
        if challenge:
            img = ((torch.roll(img,(3,5),(-2,-1)) if expert.mode == "pattern_profile" else img)+1)*0.93-1
        logits = expert.pretrain_logits(img,q)
        correct += int((logits.argmax(-1)==expert_class_targets(a,vocab)).sum()); total += len(img)
        loss_sum += float(F.cross_entropy(logits,expert_class_targets(a,vocab),reduction="sum"))
    return {"accuracy":correct/max(1,total),"ce":loss_sum/max(1,total),"n":total}


def train_stage0_experts(cfg, vocab, loaders):
    if set(loaders) != set(cfg.experts): raise ValueError("Expert/loader names differ")
    experts = make_experts(cfg,len(vocab)).to(cfg.device)
    reports = {}
    prompts = cfg.training_prompts
    null_id = vocab.stoi[NULL_VALUE]
    for name,loader in loaders.items():
        expert = experts[name]
        # Tiny training sets fit into CPU memory. No unseen samples are loaded here.
        samples = [loader.dataset[i] for i in range(len(loader.dataset))]
        if any(not parse_single_answer(name, vocab.decode(a)) for _,_,a in samples):
            raise ValueError(f"{name}: expert answers must be a known attribute value as a single word")
        opt = torch.optim.AdamW(expert.parameters(),lr=cfg.lr,weight_decay=1e-4)
        max_steps = max(cfg.expert_min_steps,cfg.epochs_expert*len(loader))
        stable = 0; best_score = float("inf"); best = None
        for step in range(1,max_steps+1):
            ids = torch.randint(len(samples),(cfg.batch,)).tolist()
            img = torch.stack([samples[i][0] for i in ids]).to(cfg.device)
            target = expert_class_targets(torch.stack([samples[i][2] for i in ids]).to(cfg.device), vocab)
            # Every class sees every question; remove fixed-prompt train/test shift.
            q = torch.stack([vocab.encode(random.choice(prompts),cfg.max_text_len) for _ in ids]).to(cfg.device)
            img = _augment_expert(img,name)
            img = torch.cat([img,torch.full_like(img[:1],-1)],0)
            q = torch.cat([q,q[:1]],0)
            target = torch.cat([target,torch.tensor([null_id],device=cfg.device)])
            expert.train(); logits = expert.pretrain_logits(img,q)
            loss = F.cross_entropy(logits,target)
            if not torch.isfinite(loss): raise RuntimeError(f"{name}: nonfinite loss")
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(expert.parameters(),1.0); opt.step()
            if step % cfg.expert_check_every == 0 or step == max_steps:
                clean = expert_accuracy(expert,loader.dataset,vocab,cfg)
                main = expert_accuracy(expert,loader.dataset,vocab,cfg,"これは何ですか？",challenge=True)
                score = clean["ce"]+main["ce"]
                if score < best_score:
                    best_score = score; best = {k:v.detach().cpu().clone() for k,v in expert.state_dict().items()}
                passed = clean["accuracy"]==1 and main["accuracy"]==1 and max(clean["ce"],main["ce"])<=cfg.expert_max_ce
                stable = stable+1 if passed else 0
                print(f"[Stage0][{name}] step={step}/{max_steps} train={clean} challenge={main}",flush=True)
                if step>=cfg.expert_min_steps and stable>=cfg.expert_stable_checks: break
        expert.load_state_dict(best); expert.eval()
        clean = expert_accuracy(expert,loader.dataset,vocab,cfg)
        main = expert_accuracy(expert,loader.dataset,vocab,cfg,"これは何ですか？",challenge=True)
        reports[name] = {"train":clean,"synthetic_challenge":main,"steps":step,
                         "passed":clean["accuracy"]==1 and main["accuracy"]==1}
    cfg.expert_training_report = reports
    return experts


def check_expert_readiness(experts,loaders,vocab,cfg):
    failed=[]
    for name in cfg.experts:
        r=expert_accuracy(experts[name],loaders[name].dataset,vocab,cfg)
        m=expert_accuracy(experts[name],loaders[name].dataset,vocab,cfg,"これは何ですか？",challenge=True)
        print(f"[expert check] {name}: train={r} synthetic_challenge={m}")
        if min(r["accuracy"],m["accuracy"])<1: failed.append(name)
    if failed: raise RuntimeError(f"Stage1 deferred: experts failed basic checks: {failed}. See expert_training_report.json; improve Stage0 first.")

def train_stage1(cfg, vocab, main_loader, experts=None):
    model = MainFlow(cfg, len(vocab)).to(cfg.device)
    if experts is not None:
        model.experts.load_state_dict(experts.state_dict())

    if cfg.freeze_experts_stage1:
        if experts is None:
            raise RuntimeError("freeze_experts_stage1=True ですが experts.pt がありません。先に --stage 0 を実行してください。")
        for p in model.experts.parameters():
            p.requires_grad_(False)
        model.experts.eval()
        print("[Stage1] experts frozen: shape/color/size/pattern/spacing")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)
    pad = vocab.stoi[vocab.PAD]

    for ep in range(cfg.epochs_stage1):
        model.train()
        if cfg.freeze_experts_stage1:
            model.experts.eval()
        tot = 0.0
        for img, q, a in main_loader:
            img, q, a = img.to(cfg.device), q.to(cfg.device), a.to(cfg.device)
            ans_in, ans_out = a[:, :-1], a[:, 1:]
            logits = model(img, q, ans_in)
            loss = ce_loss(logits, ans_out, pad)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[Stage1] ep{ep+1}/{cfg.epochs_stage1} loss={tot/len(main_loader):.4f}")
    return model


def train_stage2(cfg, vocab, route_loader, model: MainFlow):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gate = FlowXGate(cfg).to(cfg.device)
    opt = torch.optim.AdamW(gate.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs_stage2):
        gate.train()
        tot = 0.0
        n_ok, n_all = 0, 0
        for img, q, _a, target in route_loader:
            img, q, target = img.to(cfg.device), q.to(cfg.device), target.to(cfg.device)
            with torch.no_grad():
                z, _ = model.build_Z(img, q)
            logits, _ = gate(z)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
            pred = logits.argmax(dim=-1)
            n_ok += (pred == target).sum().item()
            n_all += target.numel()
        acc = n_ok / max(1, n_all)
        print(f"[Stage2] ep{ep+1}/{cfg.epochs_stage2} loss={tot/len(route_loader):.4f} question_route_acc={acc:.2%}")
    return gate


# ------------------------------------------------------------------
# Evaluation / diagnosis
# ------------------------------------------------------------------
@torch.no_grad()
def generate(model: MainFlow, vocab, img, q, cfg):
    model.eval()
    img, q = img.unsqueeze(0).to(cfg.device), q.unsqueeze(0).to(cfg.device)
    z, _ = model.build_Z(img, q)
    ids = [vocab.stoi[vocab.BOS]]
    for _ in range(cfg.max_answer_len - 1):
        ans_in = torch.tensor([ids], device=cfg.device)
        logits = model.decode_from_Z(z, ans_in)
        nxt = logits[:, -1].argmax(dim=-1).item()
        ids.append(nxt)
        if nxt == vocab.stoi[vocab.EOS]:
            break
    return vocab.decode(ids)


@torch.no_grad()
def output_ablation_report(model: MainFlow, vocab, dataset, cfg: CFG, max_samples: int = 0, dominance_ratio: float = 1.5):
    model.eval()
    cl = cfg.compressed_len
    expert_names = list(cfg.experts)
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)

    print("\n[output ablation report]")
    header = "label/question/fullLP " + " ".join([f"{e}_effect" for e in expert_names]) + " dominant expected hit"
    print(header)

    qtypes = list(expert_names) + ["main", "mixed"]
    by_type = {qt: {"n": 0, **{f"d_{e}": 0.0 for e in expert_names}} for qt in qtypes}
    n_hit = 0
    n_expected = 0
    rows = []

    for i in range(n):
        img, q, a = dataset[i]
        _, prompt, label = dataset.items[i]
        img = img.unsqueeze(0).to(cfg.device)
        q = q.unsqueeze(0).to(cfg.device)
        z, _ = model.build_Z(img, q)
        lp_full = target_content_logprob_from_Z(model, z, a, vocab)
        effects = {}
        for idx, name in enumerate(expert_names):
            z_drop = _remove_expert_tokens(z, idx, cl)
            lp_no = target_content_logprob_from_Z(model, z_drop, a, vocab)
            effects[name] = lp_full - lp_no

        dom = _dominant_from_effects(effects, dominance_ratio)
        qtype = question_type(prompt, cfg)
        expected = qtype if qtype in expert_names else "-"
        hit = ""
        if expected != "-":
            n_expected += 1
            hit = "o" if dom == expected else "x"
            n_hit += int(hit == "o")

        agg = by_type[qtype]
        agg["n"] += 1
        for name in expert_names:
            agg[f"d_{name}"] += effects[name]

        rows.append({
            "label": label,
            "prompt": prompt,
            "question_type": qtype,
            "full_logprob": lp_full,
            **{f"{name}_effect": effects[name] for name in expert_names},
            "dominant": dom,
            "expected": expected,
            "hit": hit,
        })

        effect_str = " ".join([f"{effects[e]:+7.3f}" for e in expert_names])
        print(f"{label:<50} {qtype:<8} {lp_full:+7.3f} {effect_str} {dom:<8} {expected:<8} {hit}")

    print("\n[output ablation summary]")
    for qt in qtypes:
        agg = by_type[qt]
        if not agg["n"]:
            continue
        means = " ".join([f"mean_{name}={agg[f'd_{name}']/agg['n']:+.4f}" for name in expert_names])
        print(f"  {qt:<8}: n={agg['n']:>4} {means}")
    if n_expected:
        print(f"  routing accuracy: {n_hit}/{n_expected} = {n_hit/n_expected:.2%}")
    return rows


@torch.no_grad()
def eval_unseen_main(model: MainFlow, vocab, dataset: PairDataset, cfg: CFG, max_samples: int = 0):
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    per = {f: {"ok": 0, "n": 0} for f in FACTOR_ORDER}
    joint_ok = 0
    exact_ok = 0
    rows = []

    print("\n[unseen evaluation]")
    for i in range(n):
        img, q, _a = dataset[i]
        _, prompt, gt = dataset.items[i]
        pred = generate(model, vocab, img, q, cfg)
        gt_map = parse_structured_answer(gt)
        pr_map = parse_structured_answer(pred)

        oks = factor_correctness(gt, pred)
        for f in FACTOR_ORDER:
            per[f]["ok"] += int(oks[f])
            per[f]["n"] += 1
        joint = all(oks.values())
        joint_ok += int(joint)
        exact_ok += int(pred == gt)
        rows.append({
            "index": i,
            "prompt": prompt,
            "gt": gt,
            "pred": pred,
            **{f"gt_{f}": gt_map.get(f, "") for f in FACTOR_ORDER},
            **{f"pred_{f}": pr_map.get(f, "") for f in FACTOR_ORDER},
            **{f"ok_{f}": int(oks[f]) for f in FACTOR_ORDER},
            "ok_joint": int(joint),
            "ok_exact": int(pred == gt),
        })
        print(f"{dataset.items[i][0].stem:<14} GT={gt:<70} pred={pred:<70} {'o' if joint else 'x'}")

    print("\n[unseen summary]")
    for f in FACTOR_ORDER:
        print(f"  {f:<8}: {per[f]['ok']}/{per[f]['n']} = {per[f]['ok']/max(1, per[f]['n']):.2%}")
    print(f"  {'joint':<8}: {joint_ok}/{n} = {joint_ok/max(1,n):.2%}")
    print(f"  {'exact':<8}: {exact_ok}/{n} = {exact_ok/max(1,n):.2%}")
    return rows


@torch.no_grad()
def eval_gate_routing(model: MainFlow, gate: FlowXGate, dataset: PairDataset, cfg: CFG, max_samples: int = 0):
    model.eval(); gate.eval()
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    name_to_idx = {n: i for i, n in enumerate(cfg.experts)}
    idx_to_name = {i: n for n, i in name_to_idx.items()}
    ok = 0
    blank_ok = 0
    total = 0
    by = {e: {"ok": 0, "n": 0} for e in cfg.experts}
    for i in range(n):
        img, q, _a = dataset[i]
        _, prompt, _label = dataset.items[i]
        expected = question_type(prompt, cfg)
        if expected not in name_to_idx:
            continue
        img = img.unsqueeze(0).to(cfg.device)
        q = q.unsqueeze(0).to(cfg.device)
        z, _ = model.build_Z(img, q)
        logits, _ = gate(z)
        pred = idx_to_name[int(logits.argmax(dim=-1).item())]
        z_blank, _ = model.build_Z(torch.full_like(img,-1),q)
        blank_pred = idx_to_name[int(gate(z_blank)[0].argmax(-1).item())]
        blank_ok += int(blank_pred==expected)
        total += 1
        by[expected]["n"] += 1
        if pred == expected:
            ok += 1
            by[expected]["ok"] += 1
    print("\n[gate routing eval]")
    print(f"  question routing: {ok}/{total} = {ok/max(1,total):.2%}")
    print(f"  black-image control: {blank_ok}/{total} = {blank_ok/max(1,total):.2%} (routing is not visual answer accuracy)")
    for e in cfg.experts:
        if by[e]["n"]:
            print(f"  {e:<8}: {by[e]['ok']}/{by[e]['n']} = {by[e]['ok']/by[e]['n']:.2%}")


# ------------------------------------------------------------------
# Build loaders
# ------------------------------------------------------------------
def build_loaders(cfg: CFG):
    root = Path(cfg.root)
    text_dirs = [
        cfg.dir_shape_text, cfg.dir_color_text, cfg.dir_size_text,
        cfg.dir_pattern_text, cfg.dir_spacing_text, cfg.dir_main_text,
    ]
    texts = []
    for d in text_dirs:
        p = root / d
        if not p.exists():
            continue
        for f in sorted(p.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
    vocab = WordVocab().build(texts + [NULL_VALUE])
    cfg.training_prompts = sorted({t.strip().splitlines()[0] for t in texts})

    def mk(img_d, txt_d, prompt=None, shuffle=True):
        ds = PairDataset(root / img_d, root / txt_d, vocab, cfg, fixed_prompt=prompt)
        return DataLoader(ds, batch_size=cfg.batch, shuffle=shuffle, drop_last=False)

    expert_loaders = {
        "shape": mk(cfg.dir_shape, cfg.dir_shape_text),
        "color": mk(cfg.dir_color, cfg.dir_color_text),
        "size": mk(cfg.dir_size, cfg.dir_size_text),
        "pattern": mk(cfg.dir_pattern, cfg.dir_pattern_text),
        "spacing": mk(cfg.dir_spacing, cfg.dir_spacing_text),
    }
    main_loader = mk(cfg.dir_main, cfg.dir_main_text)
    with Image.open(main_loader.dataset.items[0][0]) as sample_image:
        if min(sample_image.size) < cfg.input_size:
            print(f"[resolution] Native image {sample_image.size} < input_size={cfg.input_size}. Upsampling cannot recover lost stripe detail; regenerate v5 for a clean resolution comparison.")

    route_ds = FactorRouteDataset(root / cfg.dir_main, root / cfg.dir_main_text, vocab, cfg)
    route_loader = DataLoader(route_ds, batch_size=cfg.batch, shuffle=True, drop_last=False)

    unseen_seen = PairDataset(root / cfg.dir_unseen_seen, root / cfg.dir_unseen_seen_text, vocab, cfg)
    unseen_held = PairDataset(root / cfg.dir_unseen_heldout, root / cfg.dir_unseen_heldout_text, vocab, cfg)

    print(
        f"[data] shape={len(expert_loaders['shape'].dataset)} "
        f"color={len(expert_loaders['color'].dataset)} "
        f"size={len(expert_loaders['size'].dataset)} "
        f"pattern={len(expert_loaders['pattern'].dataset)} "
        f"spacing={len(expert_loaders['spacing'].dataset)} "
        f"main={len(main_loader.dataset)} route={len(route_ds)} vocab={len(vocab)}"
    )
    print(
        f"[data] unseen_seen={len(unseen_seen)} unseen_heldout={len(unseen_held)}"
    )
    return vocab, expert_loaders, main_loader, route_loader, unseen_seen, unseen_held


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Reproducible vocabulary/config and direct expert diagnostics (v5)
# ------------------------------------------------------------------
ARCH_KEYS = ("format_version","d_model","n_heads","enc_layers","dec_layers",
             "compressed_len","max_text_len","max_answer_len","img_size",
             "input_size","experts","expert_transforms","foreground_threshold",
             "shape_mask_value","shape_normalized_extent")


def training_fingerprint(cfg):
    h=hashlib.sha256(); root=Path(cfg.root)
    dirs=[getattr(cfg,"dir_"+e) for e in cfg.experts]+[getattr(cfg,"dir_"+e+"_text") for e in cfg.experts]+[cfg.dir_main,cfg.dir_main_text]
    for d in dirs:
        for p in sorted((root/d).iterdir()):
            if p.is_file(): h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def verify_run_manifest(ckpt,cfg,vocab,create=False):
    path=Path(ckpt,"run_manifest.json")
    obj={"architecture":{k:getattr(cfg,k) for k in ARCH_KEYS},"vocab":vocab.itos,
         "training_data_sha256":training_fingerprint(cfg)}
    obj=json.loads(json.dumps(obj))
    if path.exists():
        old=json.loads(path.read_text())
        if old != obj: raise RuntimeError("Checkpoint vocabulary/config/training-data mismatch. Use a new checkpoint directory and retrain Stage0 onward.")
    elif create:
        if any(Path(ckpt,n).exists() for n in ("experts.pt","main.pt","flowx.pt")):
            raise RuntimeError("Legacy weights found without v9 manifest. Use a new checkpoint directory.")
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    else: raise RuntimeError("Missing v9 run_manifest.json. Run Stage0 in a new directory first.")
    return obj


def load_manifest_config(ckpt,cfg):
    path=Path(ckpt,"run_manifest.json")
    if not path.exists(): raise RuntimeError("v9 requires run_manifest.json from Stage0")
    arch = json.loads(path.read_text())["architecture"]
    if arch.get("format_version") != TEXT_FORMAT_VERSION:
        raise RuntimeError("v9 text format mismatch. Use new v9 data/checkpoints and retrain Stage0.")
    for k,v in arch.items():
        setattr(cfg,k,tuple(v) if k=="experts" else v)
    return cfg


@torch.no_grad()
def audit_experts(experts,vocab,cfg,loaders,main_ds,unseen_seen,unseen_held,limit=0):
    experts.eval()
    report={"note":"Read-only evaluation; union labels are never used for optimization.","train":{},"splits":{}}
    prompts={e:loaders[e].dataset.items[0][1] for e in cfg.experts}
    for e in cfg.experts:
        report["train"][e]=expert_accuracy(experts[e],loaders[e].dataset,vocab,cfg)
    for split,ds in [("main_seen",main_ds),("union_seen_components",unseen_seen),("union_heldout_cs",unseen_held)]:
        records=[]
        for i,(_p,q,label) in enumerate(ds.items):
            if question_type(q,cfg)=="main": records.append(i)
        if limit>0:
            # Spread limited samples across the lexicographic factor grid.
            records=[records[i] for i in np.linspace(0,len(records)-1,min(limit,len(records)),dtype=int)] if records else []
        joint_correct=joint_main_correct=joint_n=0
        stat={e:{"n":0,"correct":0,"main_prompt_correct":0,"examples":[]} for e in cfg.experts}
        for start in range(0,len(records),cfg.batch):
            ids=records[start:start+cfg.batch]
            img=torch.stack([ds[i][0] for i in ids]).to(cfg.device)
            gt=[parse_structured_answer(ds.items[i][2]) for i in ids]
            predictions={}; main_predictions={}
            for e in cfg.experts:
                q=vocab.encode(prompts[e],cfg.max_text_len).to(cfg.device)[None].expand(len(ids),-1)
                qm=vocab.encode("これは何ですか？",cfg.max_text_len).to(cfg.device)[None].expand(len(ids),-1)
                pred=experts[e].pretrain_logits(img,q).argmax(-1).tolist()
                pm=experts[e].pretrain_logits(img,qm).argmax(-1).tolist()
                predictions[e]=[vocab.itos[x] for x in pred]; main_predictions[e]=[vocab.itos[x] for x in pm]
                for j,i in enumerate(ids):
                    expected=gt[j].get(e,"")
                    if not expected or expected==NULL_VALUE: continue
                    st=stat[e]; st["n"]+=1; st["correct"]+=int(vocab.itos[pred[j]]==expected)
                    st["main_prompt_correct"]+=int(vocab.itos[pm[j]]==expected)
                    if vocab.itos[pred[j]]!=expected and len(st["examples"])<15:
                        st["examples"].append({"file":ds.items[i][0].name,"gt":expected,"pred":vocab.itos[pred[j]]})
            for j in range(len(ids)):
                active=[e for e in cfg.experts if gt[j].get(e,"") not in ("",NULL_VALUE)]
                if active:
                    joint_n+=1
                    joint_correct+=int(all(predictions[e][j]==gt[j][e] for e in active))
                    joint_main_correct+=int(all(main_predictions[e][j]==gt[j][e] for e in active))
        report.setdefault("direct_expert_joint",{})[split]={"n":joint_n,"correct":joint_correct,"main_prompt_correct":joint_main_correct,
            "note":"Joint over present attributes. Union contains all five; no MainDecoder is used."}
        for e,st in stat.items():
            st["accuracy"]=st["correct"]/max(1,st["n"])
            st["main_prompt_accuracy"]=st["main_prompt_correct"]/max(1,st["n"])
        report["splits"][split]=stat
        print(f"[expert audit][{split}] "+" ".join(f"{e}={v['accuracy']:.1%}/{v['main_prompt_accuracy']:.1%}(own/main Q,n={v['n']})" for e,v in stat.items()),flush=True)
    return report


@torch.no_grad()
def eval_seen_answers(model,vocab,ds,cfg,limit=0):
    stats={}
    for i in range(len(ds) if limit<=0 else min(limit,len(ds))):
        img,q,_a=ds[i]; _path,prompt,gt=ds.items[i]
        qt=question_type(prompt,cfg); pred=generate(model,vocab,img,q,cfg)
        st=stats.setdefault(qt,{"n":0,"correct":0});st["n"]+=1;st["correct"]+=int(pred==gt)
    print("[seen free generation]",json.dumps(stats,ensure_ascii=False))
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=CFG_.root)
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--ckpt", type=str, default="ckpt_factor_v9_compact")

    ap.add_argument("--epochs-expert", type=int, default=CFG_.epochs_expert)
    ap.add_argument("--epochs-stage1", type=int, default=CFG_.epochs_stage1)
    ap.add_argument("--epochs-stage2", type=int, default=CFG_.epochs_stage2)
    ap.add_argument("--input-size", type=int, default=CFG_.input_size)
    ap.add_argument("--seed", type=int, default=CFG_.seed)
    ap.add_argument("--eval-experts", action="store_true")
    ap.add_argument("--eval-seen", action="store_true")
    ap.add_argument("--batch", type=int, default=CFG_.batch)
    ap.add_argument("--lr", type=float, default=CFG_.lr)

    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--diag-limit", type=int, default=0)
    ap.add_argument("--eval-unseen", action="store_true")
    ap.add_argument("--eval-heldout", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--eval-gate", action="store_true")

    args, _unknown = ap.parse_known_args()

    cfg = CFG_
    cfg.root = args.root
    cfg.epochs_expert = args.epochs_expert
    cfg.epochs_stage1 = args.epochs_stage1
    cfg.epochs_stage2 = args.epochs_stage2
    cfg.batch = args.batch
    cfg.lr = args.lr
    cfg.input_size = args.input_size
    cfg.seed = args.seed
    seed_all(cfg.seed)

    os.makedirs(args.ckpt, exist_ok=True)
    vocab, expert_loaders, main_loader, route_loader, unseen_seen, unseen_held = build_loaders(cfg)

    verify_run_manifest(args.ckpt,cfg,vocab,create=(args.stage==0 and not (args.eval_experts or args.eval_seen or args.eval_unseen or args.eval_heldout or args.eval_gate or args.diagnose_only)))
    main_path = os.path.join(args.ckpt, "main.pt")
    experts_path = os.path.join(args.ckpt, "experts.pt")
    flowx_path = os.path.join(args.ckpt, "flowx.pt")

    if args.eval_experts:
        experts = make_experts(cfg,len(vocab)).to(cfg.device)
        experts.load_state_dict(torch.load(experts_path,map_location=cfg.device,weights_only=True))
        report = audit_experts(experts,vocab,cfg,expert_loaders,main_loader.dataset,unseen_seen,unseen_held,args.eval_limit)
        Path(args.ckpt,"expert_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        raise SystemExit(0)

    if args.diagnose_only:
        model = MainFlow(cfg, len(vocab)).to(cfg.device)
        model.load_state_dict(torch.load(main_path, map_location=cfg.device))
        print(f"[diagnose] loaded: {main_path}")
        output_ablation_report(model, vocab, main_loader.dataset, cfg, max_samples=args.diag_limit)
        raise SystemExit(0)

    if args.eval_unseen or args.eval_heldout or args.eval_gate or args.eval_seen:
        model = MainFlow(cfg, len(vocab)).to(cfg.device)
        model.load_state_dict(torch.load(main_path, map_location=cfg.device))
        print(f"[eval] loaded: {main_path}")
        if args.eval_seen:
            eval_seen_answers(model,vocab,main_loader.dataset,cfg,args.eval_limit)
        if args.eval_unseen:
            print("\n==== unseen_union_seen_components ====")
            eval_unseen_main(model, vocab, unseen_seen, cfg, max_samples=args.eval_limit)
        if args.eval_heldout:
            print("\n==== unseen_union_heldout_cs ====")
            eval_unseen_main(model, vocab, unseen_held, cfg, max_samples=args.eval_limit)
        if args.eval_gate:
            if not os.path.exists(flowx_path):
                raise FileNotFoundError(f"flowx.pt がありません: {flowx_path}")
            gate = FlowXGate(cfg).to(cfg.device)
            gate.load_state_dict(torch.load(flowx_path, map_location=cfg.device))
            print(f"[eval] gate loaded: {flowx_path}")
            eval_gate_routing(model, gate, route_loader.dataset.base, cfg, max_samples=args.eval_limit)
        raise SystemExit(0)

    if args.stage == 0:
        experts = train_stage0_experts(cfg, vocab, expert_loaders)
        torch.save(experts.state_dict(), experts_path)
        Path(args.ckpt,"expert_training_report.json").write_text(json.dumps(cfg.expert_training_report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"[saved] {experts_path}")

    elif args.stage == 1:
        experts = None
        if os.path.exists(experts_path):
            experts = make_experts(cfg, len(vocab))
            experts.load_state_dict(torch.load(experts_path, map_location=cfg.device))
            print(f"[Stage1] pretrained experts loaded: {experts_path}")
        else:
            if cfg.freeze_experts_stage1:
                raise FileNotFoundError(
                    f"Stage1で専門家freezeを行うため experts.pt が必要です: {experts_path}\n"
                    "先に --stage 0 を実行してください。"
                )
            print("[Stage1] warning: experts.pt が無いため専門家はランダム初期化です")
        if experts is not None:
            experts = experts.to(cfg.device)
            check_expert_readiness(experts,expert_loaders,vocab,cfg)
        model = train_stage1(cfg, vocab, main_loader, experts=experts)
        torch.save(model.state_dict(), main_path)
        print(f"[saved] {main_path}")
        print("\n[quick train-split ablation]")
        output_ablation_report(model, vocab, main_loader.dataset, cfg, max_samples=min(64, len(main_loader.dataset)))

    elif args.stage == 2:
        model = MainFlow(cfg, len(vocab)).to(cfg.device)
        model.load_state_dict(torch.load(main_path, map_location=cfg.device))
        print(f"[Stage2] main loaded: {main_path}")
        gate = train_stage2(cfg, vocab, route_loader, model)
        torch.save(gate.state_dict(), flowx_path)
        print(f"[saved] {flowx_path}")
        eval_gate_routing(model, gate, route_loader.dataset.base, cfg)
