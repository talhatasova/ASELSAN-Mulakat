"""Tüm modellerin paylaştığı veri yükleme ve değerlendirme yardımcıları."""
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load():
    """(ds, label_names) döner; etiketler string IOB2 olarak da erişilebilir."""
    ds = load_from_disk(str(ROOT / "data" / "turkish-wikiNER"))
    names = ds["train"].features["tags"].feature.names
    return ds, names


def as_str(ds_split, names):
    return [r["tokens"] for r in ds_split], [[names[t] for t in r["tags"]] for r in ds_split]


def spans(tags):
    """IOB2 -> {(tip, başlangıç, bitiş)}; notebook'taki ile aynı (lenient) mantık."""
    out, cur = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        prefix, _, typ = t.partition("-")
        if cur and (prefix != "I" or typ != cur[0]):
            out.append((cur[0], cur[1], i)); cur = None
        if prefix == "B" or (prefix == "I" and cur is None):
            cur = (typ, i)
    return set(out)


def seen_unseen_recall(tokens, gold, pred, train_forms):
    """Gold varlıkları train'de görülmüş/görülmemiş diye ayırıp her grupta recall hesaplar.
    Ezberleme ile genelleme arasındaki farkı ölçmek için en dürüst kesit bu."""
    hit = {"seen": [0, 0], "unseen": [0, 0]}
    for toks, g, p in zip(tokens, gold, pred):
        ps = spans(p)
        for s in spans(g):
            key = "seen" if " ".join(toks[s[1]:s[2]]) in train_forms else "unseen"
            hit[key][1] += 1
            hit[key][0] += s in ps
    return {k: round(a / b, 4) for k, (a, b) in hit.items()}


def train_surface_forms(ds, names):
    toks, tags = as_str(ds["train"], names)
    return {" ".join(t[b:e]) for t, g in zip(toks, tags) for _, b, e in spans(g)}


def evaluate(name, tokens, gold, pred, train_forms):
    """Entity-level (strict IOB2) micro P/R/F1 + tip bazında rapor + seen/unseen recall; JSON'a yazar."""
    res = {
        "model": name,
        "precision": round(precision_score(gold, pred, mode="strict", scheme=IOB2), 4),
        "recall": round(recall_score(gold, pred, mode="strict", scheme=IOB2), 4),
        "f1": round(f1_score(gold, pred, mode="strict", scheme=IOB2), 4),
        "macro_f1": round(f1_score(gold, pred, mode="strict", scheme=IOB2, average="macro"), 4),
        "recall_by_seen": seen_unseen_recall(tokens, gold, pred, train_forms),
        "per_type": classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0),
    }
    print(classification_report(gold, pred, mode="strict", scheme=IOB2, digits=3, zero_division=0))
    print({k: v for k, v in res.items() if k != "per_type"})
    (RESULTS / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res
