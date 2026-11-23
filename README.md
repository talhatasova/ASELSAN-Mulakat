# Turkish NER on turkish-wikiNER

This is the code for the NER case study (`NER__NLP_Case_Study_v2.pdf`). It compares three models on the [turkish-nlp-suite/turkish-wikiNER](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER) dataset (19 entity types, IOB2 tags):

- **CRF baseline** with hand-crafted Turkish features.
- **BERTurk** fine-tuned for token classification.
- **BERTurk + entity-swap augmentation:** each entity is swapped for another of the same type, and the Turkish suffix is fixed to match.

## Results (test set, seed 42)

Scores are strict entity-level (seqeval, IOB2). The last two columns are recall split by whether the entity also appears in the training set.

| Model | P | R | F1 | macro-F1 | Recall (seen) | Recall (unseen) |
|---|---|---|---|---|---|---|
| CRF | .722 | .644 | .681 | .583 | .757 | .570 |
| BERTurk | .831 | .791 | **.810** | .720 | .831 | .766 |
| BERTurk + swap (p=0.3) | .828 | .767 | .796 | .719 | .802 | .747 |
| BERTurk + swap (p=0.7) | .820 | .744 | .780 | .721 | .763 | .736 |

## Project layout

```
src/
  download_data.py   download the dataset to data/turkish-wikiNER
  common.py          data loading, evaluation (seqeval + seen/unseen recall), writes JSON results
  baseline_crf.py    Section 3: CRF baseline
  train_bert.py      Section 4/5: BERTurk fine-tuning (--augment for Section 5)
  augment.py         Section 5: entity swap that fixes the suffix (run it directly for a self-check)
  compare.py         Section 6: comparison tables from reports/results/*.json
notebooks/01_eda.ipynb   Section 2: exploratory data analysis
reports/
  figures/           EDA plots
  results/           per-model JSON metrics, test predictions, training logs, LaTeX tables
  latex/             report source (ACL template; compiled on Overleaf)
```

## Reproducing the results

The original run used Windows 11, Python 3.13, and an NVIDIA RTX 4070 Laptop GPU (8 GB). All commands are run from the repository root.

### 1. Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # CUDA build
pip install -r requirements.txt
```

On a machine without a GPU, install the default `torch` build instead. Training then runs in fp32 on the CPU, which is much slower.

### 2. Data

```bash
python src/download_data.py        # saves to data/turkish-wikiNER (17,967 / 1,000 / 1,000 sentences)
```

### 3. EDA (Section 2)

Open `notebooks/01_eda.ipynb` and run all cells. The figures are written to `reports/figures/`.

### 4. Experiments

The scripts import each other, so run them from inside `src/`:

```bash
cd src
python augment.py                              # self-check for the suffix rules (prints "ok")
python baseline_crf.py                         # CRF          -> reports/results/crf.json            (~10 min on CPU)
python train_bert.py                           # BERTurk      -> reports/results/berturk.json        (~4 min on GPU)
python train_bert.py --augment                 # + swap p=0.7 -> reports/results/berturk_aug.json    (~7 min)
python train_bert.py --augment --swap-p 0.3    # + swap p=0.3 -> reports/results/berturk_aug_p0.3.json
python compare.py                              # prints the tables and writes main_table.tex / per_type_table.tex
```

On Windows, set `PYTHONIOENCODING=utf-8` if the console fails to print Turkish characters.

Each script prints a seqeval classification report and writes a JSON file to `reports/results/` containing:
- precision, recall, micro-F1 and macro-F1;
- recall split into seen and unseen entities;
- a per-type report.

BERT checkpoints go to `checkpoints/`, which is not needed for the results.

## Settings that determine the numbers

| | Setting |
|---|---|
| Split | Official HF train/validation/test. The test set is used once, at the end. |
| Metric | seqeval, `mode="strict"`, `scheme=IOB2`, entity level |
| Seen/unseen | An entity counts as "seen" if its exact surface string occurs as an entity anywhere in train |
| CRF | L-BFGS, c1 = c2 = 0.05, 150 iterations, all possible transitions |
| Encoder | `dbmdz/bert-base-turkish-cased`, linear head; only the first subword of each word is labelled (others get -100) |
| Training | 3 epochs, lr 5e-5, linear schedule, 10% warm-up, weight decay 0.01, batch size 32, fp16, max length 192; best epoch chosen by validation loss |
| Augmentation | One augmented copy of every sentence that contains an entity (17,967 → 34,038 sentences); each entity is swapped with probability p; same RNG seed (42) |
| Seed | 42 (`transformers.set_seed` + `random.seed`) |

## Notes on reproducibility

- **BERT runs may not match exactly.** Results come from a single seed, and the GPU kernels are not fully deterministic (fp16, cuDNN). A re-run can differ by about ±0.5 F1. The CRF is deterministic.
- **Differences under about 1 F1 point are within noise.** The rare types (TIME, MONEY, PERCENT, LAW) have only 4–6 test examples each, so their per-type scores are unreliable.
- **transformers v5 API.** `train_bert.py` uses `warmup_steps=0.1`, which v5 reads as a ratio; on transformers 4.x, use `warmup_ratio=0.1` instead.
- **Known limitation of the augmentation.** It does not insert the buffer consonants y/n (*Ankara'ya* vs. *Paris'e*); see the comment in `augment.py`.
