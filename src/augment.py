"""Bölüm 5 — Ek-korumalı (suffix-preserving) varlık değiştirme augmentasyonu.

Motivasyon (EDA): test varlıklarının önemli bir kısmı train'de hiç geçmiyor ve aynı yüzey biçimi birden çok
tiple etiketlenebiliyor. Model yüzey biçimini ezberlerse görülmemiş varlıklarda düşer. Çözüm: aynı tipten
başka bir mention'ı aynı bağlama koyarak modeli BAĞLAMA bakmaya zorlamak (mention replacement, Dai & Adel 2020).

Türkçe'ye özgü kısım: varlığın son token'ı 'Ankara'da' gibi ek taşıyorsa, yeni mention'a eki taşırken
ünlü uyumu ve ünsüz benzeşmesi uygulanır:  Ankara'da -> İzmir'de,  Ankara'ya -> Paris'e (kaynaştırma 'y' düşer
basit kuralla ele alınmıyor; bkz. ponytail notu), Ankara'da -> Bitlis'te.
"""
import random
import re
from collections import defaultdict

from common import spans

APOS = re.compile(r"['’]")
BACK, FRONT = set("aıou"), set("eiöü")
ROUND = set("ouöü")
HARD = set("fstkçşhp")  # fıstıkçı şahap: sert ünsüzler -> d/t benzeşmesi
VOWELS = BACK | FRONT
# Dört yönlü (ı/i/u/ü) ve iki yönlü (a/e) ünlü uyumu tabloları
FOUR = {(True, False): "ı", (False, False): "i", (True, True): "u", (False, True): "ü"}


def last_vowel(word):
    for c in reversed(word.lower().replace("İ", "i")):
        if c in VOWELS:
            return c
    return "e"  # kısaltmalar (ABD, NATO harf okunuşu) için kaba varsayılan


def harmonize(stem, suffix):
    """Ekin İLK hecesini yeni köke uyarlar. ponytail: yalnızca ilk ünlü + d/t; kaynaştırma ünsüzü (y/n)
    ve ünlüyle biten/başlayan durumlar ele alınmıyor — augmentasyonda %100 dilbilgisi gerekmiyor, gürültü kabul."""
    if not suffix:
        return suffix
    v = last_vowel(stem)
    back, rnd = v in BACK, v in ROUND
    out, done = [], False
    for i, c in enumerate(suffix):
        if i == 0 and c in "dt":
            c = "t" if stem[-1:].lower() in HARD else "d"
        if not done and c in VOWELS:
            c = ("a" if back else "e") if c in "ae" else FOUR[(back, rnd)]
            done = True
        out.append(c)
    return "".join(out)


def mention_bank(tokens, tags):
    """tip -> ek'siz mention token listeleri (son token'daki kesme+ek atılır)."""
    bank = defaultdict(list)
    for toks, tg in zip(tokens, tags):
        for typ, b, e in spans(tg):
            m = toks[b:e]
            m = m[:-1] + [APOS.split(m[-1], 1)[0]]
            if all(m):
                bank[typ].append(m)
    return bank


def swap(toks, tags, bank, rng, p=0.7):
    new_t, new_g, i = [], [], 0
    for typ, b, e in sorted(spans(tags), key=lambda s: s[1]):
        new_t += toks[i:b]; new_g += tags[i:b]
        old = toks[b:e]
        if rng.random() < p and len(bank[typ]) > 1:
            rep = list(rng.choice(bank[typ]))
            parts = APOS.split(old[-1], 1)
            if len(parts) == 2:  # orijinal ek varsa yeni köke uyarlayıp ekle
                rep[-1] = f"{rep[-1]}'{harmonize(rep[-1], parts[1])}"
        else:
            rep = old
        new_t += rep
        new_g += [f"B-{typ}"] + [f"I-{typ}"] * (len(rep) - 1)
        i = e
    return new_t + toks[i:], new_g + tags[i:]


def augment_dataset(tokens, tags, seed=42, copies=1, p=0.7):
    """Varlık içeren her cümle için `copies` adet swap'lı kopya ekler; orijinaller korunur."""
    rng = random.Random(seed)
    bank = mention_bank(tokens, tags)
    aug_t, aug_g = list(tokens), list(tags)
    for toks, tg in zip(tokens, tags):
        if any(t != "O" for t in tg):
            for _ in range(copies):
                t2, g2 = swap(toks, tg, bank, rng, p)
                aug_t.append(t2); aug_g.append(g2)
    print(f"augmentasyon: {len(tokens)} -> {len(aug_t)} cümle")
    return aug_t, aug_g


if __name__ == "__main__":
    # Hızlı kontrol: ünlü uyumu / ünsüz benzeşmesi
    assert harmonize("İzmir", "da") == "de"
    assert harmonize("Bitlis", "da") == "te"
    assert harmonize("Ankara", "nin") == "nın"
    assert harmonize("Bolu", "nin") == "nun"
    assert harmonize("Köln", "dan") == "den"
    rng = random.Random(0)
    toks = ["Mustafa", "Kemal", "Samsun'a", "çıktı", "."]
    tags = ["B-PERSON", "I-PERSON", "B-GPE", "O", "O"]
    bank = {"PERSON": [["Ali"], ["Fatma", "Yılmaz"]], "GPE": [["Bitlis"], ["İzmir"]]}
    from collections import defaultdict as dd
    for _ in range(3):
        print(swap(toks, tags, dd(list, bank), rng, p=1.0))
    print("ok")
