"""Compact Japanese noun phrases and one-word expert answers for this benchmark.

The fixed vocabulary and grammar are shared by data generation and evaluation.
No factor tags or hidden factor-ID tokens are inserted into answers.
"""
import re

TEXT_FORMAT_VERSION = "v9-compact-japanese-1"
FACTOR_ORDER = ("color", "size", "shape", "pattern", "spacing")
FACTOR_VALUES = {
    "color": ("しろい", "あかい", "あおい", "みどり", "ちゃいろい", "オリーブ"),
    "size": ("とても小さい", "小さい", "少し小さい", "普通", "少し大きい", "大きい", "とても大きい"),
    "shape": ("まる", "さんかく", "しかく", "ほし", "ごかくけい"),
    "pattern": ("ストライプ", "ボーダー"),
    "spacing": ("とても狭い", "狭い", "普通", "広い", "とても広い"),
}
SIZE_FORMS = {
    "とても小さい": "とても小さな", "小さい": "小さな", "少し小さい": "少し小さな",
    "普通": "普通の大きさの", "少し大きい": "少し大きな", "大きい": "大きな",
    "とても大きい": "とても大きな",
}
NULL_VALUE = "なし"
QUESTIONS = ("これは何ですか？", "これは何色ですか？", "これはどんな形状ですか？",
             "これはどれくらいの大きさですか？", "これはどんな模様ですか？",
             "線の間隔はどれくらいですか？")
# 普通 remains one value token; の大きさの is grammatical text. Other size
# inflections (大きな, 小さな) are visible lexical forms of the same size classes.
VALUE_WORDS = (frozenset(v for vv in FACTOR_VALUES.values() for v in vv)
               | {v for k, v in SIZE_FORMS.items() if k != "普通"} | {NULL_VALUE})
GRAMMAR_WORDS = ("の", "大きさ", "。")
LEXICON = sorted(VALUE_WORDS | set(GRAMMAR_WORDS) | set(QUESTIONS), key=lambda s: (-len(s), s))


def normalize_value(factor, value):
    if factor == "spacing" and value.startswith("間隔"):
        value = value[len("間隔"):]
    if factor == "color" and value in ("みどりの", "オリーブの"):
        value = value[:-1]
    if factor == "size":
        value = {v: k for k, v in SIZE_FORMS.items()}.get(value, value)
    if value not in FACTOR_VALUES[factor]:
        raise ValueError(f"Unknown {factor} value: {value!r}")
    return value


def single_answer(factor, value):
    return normalize_value(factor, value)


def main_answer(color, size, shape, pattern, spacing):
    v = {k: normalize_value(k, x) for k, x in zip(FACTOR_ORDER, (color, size, shape, pattern, spacing))}
    cp = v["color"] + ("の" if v["color"] in ("みどり", "オリーブ") else "")
    sp = v["spacing"] + ("の" if v["spacing"] == "普通" else "")
    return f"{cp}{sp}{v['pattern']}の{SIZE_FORMS[v['size']]}{v['shape']}。"


def tokenize(text):
    result = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        word = next((w for w in LEXICON if text.startswith(w, i)), text[i])
        result.append(word)
        i += len(word)
    return result


def parse_single_answer(factor, text):
    return text if text in FACTOR_VALUES[factor] else ""


def parse_answer(text):
    """Read the two halves around Patternの, then independent value fields.

    The short noun phrase has a fixed order: color, spacing, pattern, size,
    shape. Repeated/ambiguous values fail. This is not a general NLP parser.
    """
    out = {k: "" for k in FACTOR_ORDER}
    if "<" in text or ">" in text or not text.endswith("。"):
        return out
    tokens = tokenize(text)
    patterns = [(i, t) for i, t in enumerate(tokens) if t in FACTOR_VALUES["pattern"]]
    if len(patterns) != 1:
        return out
    pos, pattern = patterns[0]
    if tokens[pos+1:pos+2] != ["の"]:
        return out
    out["pattern"] = pattern
    left, right = tokens[:pos], tokens[pos+2:-1]
    # Each half is bounded by the pattern, so 普通 before/after it is unambiguous.
    if left and left[0] in FACTOR_VALUES["color"]:
        color = left[0]
        prefix = [color, "の"] if color in ("みどり", "オリーブ") else [color]
        if left[:len(prefix)] == prefix:
            out["color"] = color
            remainder = left[len(prefix):]
            for spacing in FACTOR_VALUES["spacing"]:
                if remainder == ([spacing, "の"] if spacing == "普通" else [spacing]):
                    out["spacing"] = spacing
    if right and right[-1] in FACTOR_VALUES["shape"]:
        out["shape"] = right[-1]
        for size, form in SIZE_FORMS.items():
            if right[:-1] == tokenize(form):
                out["size"] = size
    return out


def factor_correctness(truth, prediction):
    gt = parse_answer(truth)
    if any(not gt[f] for f in FACTOR_ORDER):
        raise ValueError(f"Invalid five-factor ground truth: {truth!r}")
    pr = parse_answer(prediction)
    return {f: bool(pr[f]) and pr[f] == gt[f] for f in FACTOR_ORDER}
