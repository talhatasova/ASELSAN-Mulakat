"""Bölüm 4/5 — BERTurk ile token sınıflandırma (+ opsiyonel ek-korumalı varlık değiştirme augmentasyonu).

Mimari kararlar:
  * dbmdz/bert-base-turkish-cased: Türkçe korpusla (~35GB) eğitilmiş, cased -> büyük harf sinyali korunuyor.
    Çok dilli mBERT/XLM-R'a göre Türkçe kelime parçalama daha verimli (daha az alt-kelime / kelime).
  * Etiket hizalama: kelimenin yalnızca İLK alt-kelimesi etiketlenir, diğerleri -100 (loss dışı).
    Türkçe'de bir kelime çok sayıda ek alt-kelimesine bölünür; hepsini etiketlemek loss'u uzun kelimelere doğru
    çarpıtır. Tahmin de ilk alt-kelimeden okunur -> kelime seviyesinde değerlendirme ile birebir uyumlu.
  * Basit lineer head (CRF head yok): BERT bağlamı geçişleri zaten iyi öğreniyor; zaman kısıtında öncelik değil.

Kullanım:  python train_bert.py                 # standart fine-tune
           python train_bert.py --augment       # Bölüm 5: ek-korumalı mention swap
"""
import argparse
import random

import numpy as np
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

from augment import augment_dataset
from common import ROOT, as_str, evaluate, load, train_surface_forms

MODEL = "dbmdz/bert-base-turkish-cased"
MAX_LEN = 192


def encode(tok, tokens, tags, label2id):
    enc = tok(tokens, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    labels = []
    for i, tag_seq in enumerate(tags):
        prev, lab = None, []
        for w in enc.word_ids(i):
            lab.append(-100 if w is None or w == prev else label2id[tag_seq[w]])
            prev = w
        labels.append(lab)
    return [{"input_ids": a, "attention_mask": m, "labels": l}
            for a, m, l in zip(enc["input_ids"], enc["attention_mask"], labels)]


def predict(trainer, tok, tokens, id2label):
    """Alt-kelime logit'lerini kelime seviyesine indirger (ilk alt-kelime). Kesilen kelimeler 'O' kalır."""
    feats = encode(tok, tokens, [["O"] * len(t) for t in tokens], {"O": 0})
    logits = trainer.predict(feats).predictions
    out = []
    for i, toks in enumerate(tokens):
        enc = tok(toks, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
        pred, prev = ["O"] * len(toks), None
        for j, w in enumerate(enc.word_ids()):
            if w is not None and w != prev:
                pred[w] = id2label[int(logits[i][j].argmax())]
            prev = w
        out.append(pred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--swap-p", type=float, default=0.7)
    args = ap.parse_args()
    set_seed(args.seed); random.seed(args.seed)
    name = (f"berturk_aug_p{args.swap_p}" if args.swap_p != 0.7 else "berturk_aug") if args.augment else "berturk"

    ds, names = load()
    label2id = {l: i for i, l in enumerate(names)}
    id2label = dict(enumerate(names))
    Xtr, ytr = as_str(ds["train"], names)
    if args.augment:
        Xtr, ytr = augment_dataset(Xtr, ytr, seed=args.seed, p=args.swap_p)
    Xva, yva = as_str(ds["validation"], names)
    Xte, yte = as_str(ds["test"], names)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=len(names),
                                                            id2label=id2label, label2id=label2id)
    targs = TrainingArguments(
        output_dir=str(ROOT / "checkpoints" / name), num_train_epochs=args.epochs,
        learning_rate=5e-5, warmup_steps=0.1,  # transformers v5: <1 ise oran olarak yorumlanıyor weight_decay=0.01, lr_scheduler_type="linear",
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(), eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=100, report_to=[], seed=args.seed, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=encode(tok, Xtr, ytr, label2id),
                      eval_dataset=encode(tok, Xva, yva, label2id),
                      data_collator=DataCollatorForTokenClassification(tok))
    trainer.train()
    trainer.save_model(str(ROOT / "checkpoints" / name / "best"))

    pred = predict(trainer, tok, Xte, id2label)
    evaluate(name, Xte, yte, pred, train_surface_forms(ds, names))
    np.save(ROOT / "reports" / "results" / f"{name}_test_pred.npy", np.array(pred, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
