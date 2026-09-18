#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9 compact-answer joint-main dataset with minimal canonical expert examples. Use the matching v9 compact-answer training scripts.

All main images contain Color, Shape, Size, Pattern and Spacing.
Split: main=1400; unseen full tuples with seen color/shape pairs=350;
held-out color/shape pairs=350. Main_cont contains 6 Q/A copies per image.
Shape/Color/Size and the joint-main split/rasterizer are unchanged.
Pattern has 2 canonical examples; Spacing has 5 canonical examples.
No file from either evaluation split is used to generate main_cont.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
from factor_text_v9 import main_answer, single_answer, parse_answer, TEXT_FORMAT_VERSION


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_OUTPUT = "データを作成するディレクトリのパス"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_SUPERSAMPLE = 4
DEFAULT_BACKGROUND = (0, 0, 0)
DEFAULT_SHAPE_GRAY = (220, 220, 220)
DEFAULT_DARK_GRAY = (72, 72, 72)
DEFAULT_SHAPE_CANONICAL_SIZE = "normal"
DEFAULT_SIZE_SHAPE_KEY = "circle"
EXPERT_PHASE = 0.5
PATTERN_EXPERT_SPACING = "normal"
SPACING_EXPERT_PATTERN = "stripe"
NONE_TOKEN = "なし"

QUESTION_MAIN = "これは何ですか？"
QUESTION_COLOR = "これは何色ですか？"
QUESTION_SHAPE = "これはどんな形状ですか？"
QUESTION_SIZE = "これはどれくらいの大きさですか？"
QUESTION_PATTERN = "これはどんな模様ですか？"
QUESTION_SPACING = "線の間隔はどれくらいですか？"

SPECIAL_TOKENS = []  # No visible answer tags; model control tokens are internal.


@dataclass(frozen=True)
class ColorSpec:
    key: str
    jp: str
    rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class ShapeSpec:
    key: str
    jp: str


@dataclass(frozen=True)
class SizeSpec:
    key: str
    jp: str
    scale: float


@dataclass(frozen=True)
class PatternSpec:
    key: str
    jp: str
    orientation: str  # vertical or horizontal


@dataclass(frozen=True)
class SpacingSpec:
    key: str
    jp: str
    # Period as a fraction of the LOCAL normalized pattern extent.
    # Smaller = denser/narrower spacing.
    period_frac: float


COLORS: List[ColorSpec] = [
    ColorSpec("white", "しろい", (255, 255, 255)),
    ColorSpec("red", "あかい", (230, 48, 48)),
    ColorSpec("blue", "あおい", (48, 108, 230)),
    ColorSpec("green", "みどりの", (40, 190, 70)),
    ColorSpec("brown", "ちゃいろい", (150, 96, 48)),
    ColorSpec("olive", "オリーブの", (128, 128, 0)),
]

SHAPES: List[ShapeSpec] = [
    ShapeSpec("circle", "まる"),
    ShapeSpec("triangle", "さんかく"),
    ShapeSpec("square", "しかく"),
    ShapeSpec("star", "ほし"),
    ShapeSpec("pentagon", "ごかくけい"),
]

SIZES: List[SizeSpec] = [
    SizeSpec("very_small", "とても小さい", 0.26),
    SizeSpec("small", "小さい", 0.34),
    SizeSpec("slightly_small", "少し小さい", 0.42),
    SizeSpec("normal", "普通", 0.50),
    SizeSpec("slightly_large", "少し大きい", 0.58),
    SizeSpec("large", "大きい", 0.66),
    SizeSpec("very_large", "とても大きい", 0.74),
]

# Japanese convention used here:
#   ストライプ = vertical stripes
#   ボーダー     = horizontal stripes
PATTERNS: List[PatternSpec] = [
    PatternSpec("stripe", "ストライプ", "vertical"),
    PatternSpec("border", "ボーダー", "horizontal"),
]

# Internal legacy class names are converted to ordinary wording in answers.
SPACINGS: List[SpacingSpec] = [
    SpacingSpec("very_narrow", "間隔とても狭い", 0.080),
    SpacingSpec("narrow", "間隔狭い", 0.110),
    SpacingSpec("normal", "間隔普通", 0.155),
    SpacingSpec("wide", "間隔広い", 0.220),
    SpacingSpec("very_wide", "間隔とても広い", 0.300),
]

# Same 5 held-out color-shape pairs as v3 for direct comparability.
# Joint main also excludes them from training.
HELDOUT_COLOR_SHAPE_PAIRS = {
    ("white", "circle"),
    ("red", "triangle"),
    ("blue", "square"),
    ("green", "star"),
    ("brown", "pentagon"),
}


# --------------------------------------------------------------------------------------
# CLI / filesystem
# --------------------------------------------------------------------------------------


DATASET_VERSION = "v9_compact_minimal_experts_joint_main_split_1400_350_350"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate all-five-factor main data; keep both evaluation splits unseen")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--overwrite", action="store_true", help="Replace only an existing dataset created by this exact generator version")
    args, unknown = p.parse_known_args(argv)
    # Colab injects -f <kernel.json>; allow this one known notebook argument only.
    if unknown and not (len(unknown)==2 and unknown[0]=="-f"):
        p.error("unrecognized arguments: " + " ".join(unknown))
    if args.image_size < 256: p.error("Use image-size >= 256 to preserve the narrowest stripes")
    return args


SUBDIRS = [
    "color", "color_text", "shape", "shape_text", "size", "size_text",
    "pattern", "pattern_text", "spacing", "spacing_text",
    "main", "main_text", "main_cont", "main_text_cont",
    "unseen_union_seen_components", "unseen_union_seen_components_text",
    "unseen_union_heldout_cs", "unseen_union_heldout_cs_text", "metadata",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {root}. Choose a new --output directory.")
        config = root / "metadata" / "config.json"
        if not config.exists() or json.loads(config.read_text(encoding="utf-8")).get("version") != DATASET_VERSION:
            raise RuntimeError("Refusing to overwrite a different dataset. Choose a new --output directory.")
        for name in SUBDIRS:
            if (root/name).exists(): shutil.rmtree(root/name)
    ensure_dir(root)
    for name in SUBDIRS: ensure_dir(root/name)


def write_two_line_text(path: Path, question: str, answer: str) -> None:
    path.write_text(f"{question}\n{answer}\n", encoding="utf-8")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------------------
# Geometry / rendering
# --------------------------------------------------------------------------------------


def regular_polygon_points(cx: float, cy: float, radius: float, sides: int,
                           rotation_deg: float) -> List[Tuple[float, float]]:
    pts = []
    rot = math.radians(rotation_deg)
    for i in range(sides):
        t = rot + 2.0 * math.pi * i / sides
        pts.append((cx + radius * math.cos(t), cy + radius * math.sin(t)))
    return pts


def star_points(cx: float, cy: float, outer_r: float, inner_r: float,
                rotation_deg: float) -> List[Tuple[float, float]]:
    pts = []
    rot = math.radians(rotation_deg)
    for i in range(10):
        r = outer_r if i % 2 == 0 else inner_r
        t = rot + 2.0 * math.pi * i / 10.0
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


def draw_shape_mask(shape_key: str, image_size: int, scale: float,
                    supersample: int = DEFAULT_SUPERSAMPLE) -> Image.Image:
    big = image_size * supersample
    mask = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(mask)
    cx = big / 2.0
    cy = big / 2.0
    radius = big * scale / 2.0

    if shape_key == "circle":
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=255)
    elif shape_key == "triangle":
        draw.polygon(regular_polygon_points(cx, cy, radius, 3, -90), fill=255)
    elif shape_key == "square":
        draw.polygon(regular_polygon_points(cx, cy, radius, 4, 45), fill=255)
    elif shape_key == "pentagon":
        draw.polygon(regular_polygon_points(cx, cy, radius, 5, -90), fill=255)
    elif shape_key == "star":
        draw.polygon(star_points(cx, cy, radius, radius * 0.45, -90), fill=255)
    else:
        raise ValueError(f"unknown shape_key: {shape_key}")
    return mask


def draw_solid_shape(shape_key: str, fill_rgb: Tuple[int, int, int], image_size: int,
                     scale: float, supersample: int = DEFAULT_SUPERSAMPLE) -> Image.Image:
    big = image_size * supersample
    mask = draw_shape_mask(shape_key, image_size, scale, supersample)
    fg = Image.new("RGB", (big, big), fill_rgb)
    bg = Image.new("RGB", (big, big), DEFAULT_BACKGROUND)
    out = Image.composite(fg, bg, mask)
    return out.resize((image_size, image_size), Image.Resampling.LANCZOS)


def darken(rgb: Tuple[int, int, int], factor: float = 0.42) -> Tuple[int, int, int]:
    # Keep a small floor so dark bands are never identical to the black background.
    return tuple(max(18, min(255, int(round(v * factor)))) for v in rgb)


def lighten(rgb: Tuple[int, int, int], factor: float = 1.05) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(v * factor)))) for v in rgb)


def stable_phase01(*parts: str) -> float:
    raw = "|".join(parts).encode("utf-8")
    h = hashlib.sha256(raw).digest()
    n = int.from_bytes(h[:8], "big")
    return (n % 1_000_003) / 1_000_003.0


def make_pattern_rgb(width: int, height: int,
                     pattern: PatternSpec, spacing: SpacingSpec,
                     base_rgb: Tuple[int, int, int], phase01: float) -> Image.Image:
    """Create pattern in LOCAL normalized coordinates.

    period_frac is relative to width for vertical stripes and height for horizontal
    borders. Thus resizing/cropping a shape does not turn absolute Size into Spacing.
    """
    bright = lighten(base_rgb)
    dark = darken(base_rgb)
    img = Image.new("RGB", (width, height), bright)
    draw = ImageDraw.Draw(img)

    extent = width if pattern.orientation == "vertical" else height
    period = max(2, int(round(extent * spacing.period_frac)))
    half = max(1, period // 2)
    offset = int(round(phase01 * period)) % period

    if pattern.orientation == "vertical":
        start = -period + offset
        x = start
        band_idx = 0
        while x < width + period:
            if band_idx % 2 == 0:
                draw.rectangle([x, 0, x + half - 1, height], fill=dark)
            x += half
            band_idx += 1
    elif pattern.orientation == "horizontal":
        start = -period + offset
        y = start
        band_idx = 0
        while y < height + period:
            if band_idx % 2 == 0:
                draw.rectangle([0, y, width, y + half - 1], fill=dark)
            y += half
            band_idx += 1
    else:
        raise ValueError(pattern.orientation)
    return img


def render_pattern_canvas(color_rgb: Tuple[int, int, int], pattern: PatternSpec,
                          spacing: SpacingSpec, image_size: int, phase01: float,
                          supersample: int = DEFAULT_SUPERSAMPLE) -> Image.Image:
    big = image_size * supersample
    p = make_pattern_rgb(big, big, pattern, spacing, color_rgb, phase01)
    return p.resize((image_size, image_size), Image.Resampling.LANCZOS)


def render_patterned_shape(shape: ShapeSpec, size: SizeSpec, color: ColorSpec,
                           pattern: PatternSpec, spacing: SpacingSpec,
                           image_size: int, phase01: float,
                           supersample: int = DEFAULT_SUPERSAMPLE) -> Image.Image:
    """Pattern is generated in the shape bounding box's normalized coordinate frame."""
    big = image_size * supersample
    mask = draw_shape_mask(shape.key, image_size, size.scale, supersample)
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError("empty shape mask")
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1-x0), max(1, y1-y0)

    local = make_pattern_rgb(bw, bh, pattern, spacing, color.rgb, phase01)
    patterned = Image.new("RGB", (big, big), DEFAULT_BACKGROUND)
    patterned.paste(local, (x0, y0))
    bg = Image.new("RGB", (big, big), DEFAULT_BACKGROUND)
    out = Image.composite(patterned, bg, mask)
    return out.resize((image_size, image_size), Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------------------
# Answers / rows
# --------------------------------------------------------------------------------------


def structured_answer(color, size, shape, pattern, spacing):
    return main_answer(color, size, shape, pattern, spacing)


def heldout_cs(color_key: str, shape_key: str) -> bool:
    return (color_key, shape_key) in HELDOUT_COLOR_SHAPE_PAIRS


# --------------------------------------------------------------------------------------
# Expert datasets
# --------------------------------------------------------------------------------------


def generate_color_expert(root: Path, image_size: int) -> List[dict]:
    rows = []
    for i, c in enumerate(COLORS):
        stem = f"color_{i:04d}"
        Image.new("RGB", (image_size, image_size), c.rgb).save(root / "color" / f"{stem}.png")
        write_two_line_text(root / "color_text" / f"{stem}.txt", QUESTION_COLOR, single_answer("color", c.jp))
        rows.append({"stem": stem, "color_key": c.key, "color_text": c.jp, "answer": single_answer("color", c.jp)})
    return rows


def generate_shape_expert(root: Path, image_size: int) -> List[dict]:
    rows = []
    normal = next(z for z in SIZES if z.key == DEFAULT_SHAPE_CANONICAL_SIZE)
    for i, s in enumerate(SHAPES):
        stem = f"shape_{i:04d}"
        draw_solid_shape(s.key, DEFAULT_SHAPE_GRAY, image_size, normal.scale).save(root / "shape" / f"{stem}.png")
        write_two_line_text(root / "shape_text" / f"{stem}.txt", QUESTION_SHAPE, single_answer("shape", s.jp))
        rows.append({"stem": stem, "shape_key": s.key, "shape_text": s.jp, "answer": single_answer("shape", s.jp)})
    return rows


def generate_size_expert(root: Path, image_size: int) -> List[dict]:
    rows = []
    for i, z in enumerate(SIZES):
        stem = f"size_{i:04d}"
        draw_solid_shape(DEFAULT_SIZE_SHAPE_KEY, DEFAULT_SHAPE_GRAY, image_size, z.scale).save(root / "size" / f"{stem}.png")
        write_two_line_text(root / "size_text" / f"{stem}.txt", QUESTION_SIZE, single_answer("size", z.jp))
        rows.append({"stem": stem, "size_key": z.key, "size_text": z.jp, "answer": single_answer("size", z.jp)})
    return rows


def generate_pattern_spacing_experts(root: Path, image_size: int) -> Tuple[List[dict], List[dict]]:
    """Vary only the target attribute; keep the other factors canonical."""
    pattern_rows: List[dict] = []
    spacing_rows: List[dict] = []
    fixed_spacing = next(d for d in SPACINGS if d.key == PATTERN_EXPERT_SPACING)
    fixed_pattern = next(p for p in PATTERNS if p.key == SPACING_EXPERT_PATTERN)
    for name, pairs, question, rows in (
        ("pattern", [(p, fixed_spacing) for p in PATTERNS], QUESTION_PATTERN, pattern_rows),
        ("spacing", [(fixed_pattern, d) for d in SPACINGS], QUESTION_SPACING, spacing_rows),
    ):
        for i, (p, d) in enumerate(pairs):
            stem = f"{name}_{i:04d}"
            answer = single_answer(name, p.jp if name == "pattern" else d.jp)
            render_pattern_canvas(DEFAULT_SHAPE_GRAY, p, d, image_size, EXPERT_PHASE).save(
                root / name / f"{stem}.png")
            write_two_line_text(root / (name + "_text") / f"{stem}.txt", question, answer)
            rows.append({"stem": stem, "pattern_key": p.key, "pattern_text": p.jp,
                         "spacing_key": d.key, "spacing_text": d.jp,
                         "phase": EXPERT_PHASE, "answer": answer})
    return pattern_rows, spacing_rows


# --------------------------------------------------------------------------------------
# Main training families
# --------------------------------------------------------------------------------------


def factor_key(row):
    return tuple(row[name+"_key"] for name in ("color","shape","size","pattern","spacing"))


def choose_split(ci, si, zi, pi, di):
    if heldout_cs(COLORS[ci].key, SHAPES[si].key):
        return "unseen_union_heldout_cs"
    # Each seen C/S pair, size and orientation has four train spacings and
    # exactly one held-out spacing. No complete five-factor tuple is shared.
    if di == (ci+si+zi+pi) % len(SPACINGS):
        return "unseen_union_seen_components"
    return "main"


def generate_joint(root, image_size):
    rows = {name: [] for name in ("main","unseen_union_seen_components","unseen_union_heldout_cs")}
    cont_rows=[]
    for ci,c in enumerate(COLORS):
        for si,s in enumerate(SHAPES):
            for zi,z in enumerate(SIZES):
                for pi,p in enumerate(PATTERNS):
                    for di,d in enumerate(SPACINGS):
                        split=choose_split(ci,si,zi,pi,di)
                        prefix={"main":"j","unseen_union_seen_components":"us","unseen_union_heldout_cs":"uh"}[split]
                        stem=f"{prefix}_{len(rows[split]):05d}"
                        # Same rendering/phase rule as v5 for a given tuple.
                        phase_name="union_plus_heldout_color_shape" if heldout_cs(c.key,s.key) else "union_only_seen_components"
                        phase=stable_phase01(phase_name,c.key,s.key,z.key,p.key,d.key)
                        image=render_patterned_shape(s,z,c,p,d,image_size,phase)
                        answer=structured_answer(c.jp,z.jp,s.jp,p.jp,d.jp)
                        textdir="main_text" if split=="main" else split+"_text"
                        image_path=root/split/(stem+".png")
                        image.save(image_path)
                        write_two_line_text(root/textdir/(stem+".txt"),QUESTION_MAIN,answer)
                        row={"stem":stem,"split":split,"color_key":c.key,"color_text":c.jp,
                             "shape_key":s.key,"shape_text":s.jp,"size_key":z.key,"size_text":z.jp,
                             "pattern_key":p.key,"pattern_text":p.jp,"spacing_key":d.key,"spacing_text":d.jp,
                             "phase":phase,"answer":answer}
                        rows[split].append(row)
                        if split=="main":
                            specs=[("main",QUESTION_MAIN,answer),("color",QUESTION_COLOR,c.jp),
                                   ("shape",QUESTION_SHAPE,s.jp),("size",QUESTION_SIZE,z.jp),
                                   ("pattern",QUESTION_PATTERN,p.jp),("spacing",QUESTION_SPACING,d.jp)]
                            for suffix,question,label in specs:
                                cpstem=stem+"_"+suffix
                                shutil.copyfile(image_path,root/"main_cont"/(cpstem+".png"))
                                label = label if suffix == "main" else single_answer(suffix, label)
                                write_two_line_text(root/"main_text_cont"/(cpstem+".txt"),question,label)
                                cont_rows.append(dict(row,stem=cpstem,base_stem=stem,qa_type=suffix,question=question,answer=label))
        print(f"[render] color {ci+1}/{len(COLORS)} complete",flush=True)
    return rows,cont_rows


def validate_split(rows,cont_rows):
    sets={name:set(map(factor_key,rr)) for name,rr in rows.items()}
    names=list(sets)
    for i,a in enumerate(names):
        assert len(sets[a])==len(rows[a]),f"duplicate tuple in {a}"
        for b in names[i+1:]: assert sets[a].isdisjoint(sets[b]),f"leakage: {a}/{b}"
    assert [len(rows[n]) for n in names]==[1400,350,350]
    train=rows["main"]
    train_cs={(r["color_key"],r["shape_key"]) for r in train}
    assert train_cs.isdisjoint(HELDOUT_COLOR_SHAPE_PAIRS)
    assert len(train_cs)==25
    assert {(r["color_key"],r["shape_key"]) for r in rows["unseen_union_seen_components"]} <= train_cs
    assert {(r["color_key"],r["shape_key"]) for r in rows["unseen_union_heldout_cs"]} == HELDOUT_COLOR_SHAPE_PAIRS
    assert len(cont_rows)==6*len(train)
    assert all(all(parse_answer(r["answer"]).values()) for r in train)
    assert all("<" not in r["answer"] for r in cont_rows)
    assert all(factor_key(r) in sets["main"] for r in cont_rows)
    # Every expert target value is also represented in joint main training.
    for name,specs in [("color",COLORS),("shape",SHAPES),("size",SIZES),("pattern",PATTERNS),("spacing",SPACINGS)]:
        assert {r[name+"_key"] for r in train} == {v.key for v in specs}
    return {"tuple_overlap":0,"train_heldout_color_shape_overlap":0,
            "main":1400,"main_cont":8400,"route":7000,
            "unseen_union_seen_components":350,"unseen_union_heldout_cs":350}


def write_joint_metadata(root,args,expert_rows,rows,cont_rows,validation):
    meta=root/"metadata"
    for name,rr in expert_rows.items(): write_csv(meta/(name+".csv"),rr)
    for name,rr in rows.items(): write_csv(meta/(name+".csv"),rr)
    write_csv(meta/"main_cont.csv",cont_rows)
    config={"version":DATASET_VERSION,"answer_format":TEXT_FORMAT_VERSION,"image_size":args.image_size,"expert_phases":1,
            "expert_protocol":{"pattern_spacing_fixed":PATTERN_EXPERT_SPACING,
                               "spacing_pattern_fixed":SPACING_EXPERT_PATTERN,
                               "phase_fixed":EXPERT_PHASE,"canvas":"full gray texture",
                               "training_augmentation":"controlled by the model script, not this generator"},
            "training_factor_families":{"joint":["color","shape","size","pattern","spacing"]},
            "heldout_color_shape_pairs":sorted(map(list,HELDOUT_COLOR_SHAPE_PAIRS)),
            "split_rule":"heldout C/S first; otherwise hold out spacing index (color_i+shape_i+size_i+pattern_i) mod 5",
            "evaluation_meaning":{
                "unseen_union_seen_components":"unseen full five-factor tuples; C/S pairs are seen in main",
                "unseen_union_heldout_cs":"C/S pairs never used in main; all sizes/patterns/spacings evaluated"},
            "questions":{"main":QUESTION_MAIN,"color":QUESTION_COLOR,"shape":QUESTION_SHAPE,
                         "size":QUESTION_SIZE,"pattern":QUESTION_PATTERN,"spacing":QUESTION_SPACING},
            "colors":[asdict(x) for x in COLORS],"shapes":[asdict(x) for x in SHAPES],
            "sizes":[asdict(x) for x in SIZES],"patterns":[asdict(x) for x in PATTERNS],
            "spacings":[asdict(x) for x in SPACINGS],"special_tokens":SPECIAL_TOKENS,
            "counts":dict(validation,**{k:len(v) for k,v in expert_rows.items()}),
            "rendering":{"background":DEFAULT_BACKGROUND,"supersample":DEFAULT_SUPERSAMPLE,
                         "spacing_coordinate_system":"relative to shape bounding box; unchanged from v5"},
            "compatibility":"Use matching v9 compact-answer training scripts with a new --root AND a new --ckpt; retrain Stage0 onward"}
    (meta/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    (meta/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    (meta/"summary.txt").write_text(
        "ALL FIVE FACTORS COEXIST IN MAIN; PLAIN JAPANESE ANSWERS\n"
        "main=1400; main_cont=8400; route=7000\n"
        "unseen full tuples (seen C/S pairs)=350; unseen C/S pairs=350\n"
        "Experts: shape=5 color=6 size=7 pattern=2 spacing=5.\n"
        "Pattern: normal spacing fixed. Spacing: stripe direction fixed. Phase=0.5.\n"
        "Joint-main and evaluation split/rendering unchanged from v6.\n"
        "Model-side online augmentation is not disabled by this generator.\n"
        "Neither evaluation split overlaps main by a full factor tuple.\n",
        encoding="utf-8")
    # Compact preview of actual MAIN examples; all show five factors.
    thumbs=[root/"main"/(rows["main"][i]["stem"]+".png") for i in (0,97,279,514,861,1229)]
    canvas=Image.new("RGB",(args.image_size*3,args.image_size*2),(24,24,24))
    for i,p in enumerate(thumbs):
        with Image.open(p) as im: canvas.paste(im,((i%3)*args.image_size,(i//3)*args.image_size))
    canvas.save(meta/"preview.png")


def main(argv=None):
    args=parse_args(argv);root=Path(args.output).expanduser()
    reset_output(root,args.overwrite)
    print(f"[v9 compact answers / minimal experts / joint main] output={root}",flush=True)
    expert_rows={"color":generate_color_expert(root,args.image_size),
                 "shape":generate_shape_expert(root,args.image_size),
                 "size":generate_size_expert(root,args.image_size)}
    expert_rows["pattern"],expert_rows["spacing"]=generate_pattern_spacing_experts(root,args.image_size)
    rows,cont_rows=generate_joint(root,args.image_size)
    validation=validate_split(rows,cont_rows)
    write_joint_metadata(root,args,expert_rows,rows,cont_rows,validation)
    print(json.dumps(dict(validation,experts={k:len(v) for k,v in expert_rows.items()}),ensure_ascii=False,indent=2),flush=True)
    print("Use --root above and a NEW --ckpt with the matching v9 compact-answer model scripts. Done.")


if __name__=="__main__": main()
