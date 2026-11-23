"""Bölüm 3 — Geleneksel baseline: el yapımı özelliklerle lineer-zincir CRF.

Neden CRF: token sınıflandırıcıdan (LogReg/SVM) farklı olarak etiket geçişlerini (B-X -> I-X) modelliyor,
böylece geçersiz IOB dizileri ve çok token'lı varlıkların kopması büyük ölçüde engelleniyor.

Özellik tasarımı Türkçe'nin eklemeli yapısı üzerine kurulu:
  * Özel isme gelen çekim eki kesme ile ayrılır (Ankara'da). Kök (Ankara) ile ek (da) ayrı özellik:
    kök sözlük-benzeri bir sinyal, ekin varlığı ise "bu bir özel isim" demenin neredeyse kesin işareti.
  * Büyük harf güçlü ama cümle başında anlamsız -> 'title & not BOS' ayrı özellik.
  * Kelime şekli (Xxxx, dddd, XX) sayısal tipleri (DATE/CARDINAL/PERCENT/MONEY) ayırmak için.
  * Kök önek/sonekleri: -oğlu, -lar, -ler, -spor, -iye gibi morfolojik ipuçları.
  * ±2 pencere bağlam: "Nehri", "Üniversitesi", "Savaşı" gibi baş kelimeler tipi belirler.
"""
import re
import time

import sklearn_crfsuite

from common import as_str, evaluate, load, train_surface_forms

APOS = re.compile(r"['’]")


def shape(w):
    s = re.sub(r"[A-ZÇĞİÖŞÜ]", "X", w)
    s = re.sub(r"[a-zçğıöşü]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)  # tekrarları kısalt: Xxxxxx -> Xxx


def word_feats(w, p=""):
    stem, suf = (APOS.split(w, 1) + [""])[:2]
    low = stem.lower()
    return {
        p + "low": w.lower(), p + "stem": low, p + "suf": suf.lower(), p + "has_apos": bool(suf),
        p + "title": w[:1].isupper(), p + "upper": w.isupper() and len(w) > 1,
        p + "digit": any(c.isdigit() for c in w), p + "shape": shape(stem),
    }


def sent2feats(toks):
    out = []
    for i, w in enumerate(toks):
        f = {"bias": 1.0, "BOS": i == 0, "EOS": i == len(toks) - 1, **word_feats(w)}
        stem = APOS.split(w, 1)[0].lower()
        for k in (2, 3, 4):
            f[f"pre{k}"], f[f"suf{k}"] = stem[:k], stem[-k:]
        f["title_mid"] = w[:1].isupper() and i > 0
        f["len"] = min(len(stem), 12)
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(toks):
                f.update(word_feats(toks[j], f"{off}:"))
            else:
                f[f"{off}:PAD"] = True
        out.append(f)
    return out


def main():
    ds, names = load()
    Xtr, ytr = as_str(ds["train"], names)
    Xte, yte = as_str(ds["test"], names)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.05, max_iterations=150,
                               all_possible_transitions=True)
    t = time.time()
    crf.fit([sent2feats(s) for s in Xtr], ytr)
    print(f"train {time.time() - t:.0f}s")
    pred = crf.predict([sent2feats(s) for s in Xte])
    evaluate("crf", Xte, yte, pred, train_surface_forms(ds, names))

    # Yorumlanabilirlik: modelin en çok güvendiği özellikler (rapor için)
    top = sorted(crf.state_features_.items(), key=lambda kv: -kv[1])[:25]
    for (feat, lab), w in top:
        print(f"{w:7.2f}  {lab:12s} {feat}")


if __name__ == "__main__":
    main()
