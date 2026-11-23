# Turkish Named Entity Recognition on turkish-wikiNER

## 1. Literature Review & Problem Definition

NER is the task of finding text spans that refer to entities and labelling each span with a type such as PERSON, ORG or GPE. It is usually cast as sequence labelling with BIO/IOB2 tags.

**Evolution of NER methods.**
- **Rule and gazetteer systems (MUC era).** High precision, but brittle and expensive to port to new domains.
- **Feature-based statistical models.** HMM, MaxEnt and especially linear-chain CRFs (Lafferty et al., 2001) using hand-crafted orthographic, affix and context features. Strong baselines that are still used in low-resource settings.
- **Neural sequence models.** BiLSTM-CRF with character embeddings (Lample et al., 2016; Ma & Hovy, 2016) learns features end to end.
- **Pretrained contextual encoders.** BERT and its relatives (Devlin et al., 2019), multilingual XLM-R, and language-specific models such as BERTurk (Schweter, 2020) are fine-tuned with a token-classification head. This is the current standard and reaches roughly 92–95 F1 on Turkish news NER.
- **LLMs.** Zero/few-shot extraction by prompting (e.g. GPT-NER, UniversalNER). Flexible for new types, but on fixed label sets these models still trail fine-tuned encoders, and they are costly at inference.

**Core challenges, and how they are addressed.**
- **Ambiguity of boundaries and types.** "Türkiye" can be a GPE, or part of an ORG name. Contextual encoders address this by reading the whole sentence.
- **Unseen / rare entities.** Long-tail types such as LAW and LANGUAGE have few examples. Remedies are subword models, character features, data augmentation and gazetteers.
- **Label noise.** Silver-standard or automatically projected annotations introduce errors. Remedies are noise-robust training and confident-learning style relabelling.
- **Domain shift.** A model trained on Wikipedia degrades on social media or military reports. Remedies are domain-adaptive pretraining and active learning.

**What is specific to Turkish.**
- Turkish is agglutinative, so one proper noun appears in many inflected forms (Ankara, Ankara'da, Ankara'nın, Ankara'dan). This sparsity hurts word-level models.
- Orthography helps: inflectional suffixes on proper nouns are separated with an apostrophe, and capitalisation marks proper nouns. Both signals break at sentence starts and in informal text.
- Word order is free (SOV), so useful context can appear on either side of a mention.

**Task.** Word-level IOB2 tagging with 18 OntoNotes-style types (+O) on turkish-wikiNER. The primary metric is strict entity-level micro-F1 (seqeval, IOB2). Macro-F1 is reported to expose rare types, and recall is split into entities **seen vs. unseen in train** to separate memorisation from generalisation.

## 2. Exploratory Data Analysis
See `notebooks/01_eda.ipynb`. Key findings: _(fill in from notebook outputs — split sizes 17,967 / 1,000 / 1,000 sentences; type distribution heavily skewed; % unseen test entities; ambiguous surface forms; label-noise examples such as "meydana gelen depremde" tagged EVENT)._

## 3. Baseline: Feature-based CRF
`src/baseline_crf.py`. The model is a linear-chain CRF (L-BFGS, L1 = L2 = 0.05) trained on whitespace tokens as given. No lowercasing or stemming is applied globally, because case is a key signal. Instead, both raw and normalised views are exposed as features:

| Feature group | Why |
|---|---|
| lowercase word, **apostrophe-split stem + suffix** | Merges Ankara / Ankara'da / Ankara'nın into one stem; the presence of a suffix is a strong proper-noun cue |
| title-case, all-caps, `title & not sentence-initial` | Capitalisation is informative except at BOS |
| word shape (Xxx, dddd, XX) | Separates DATE, CARDINAL, PERCENT and MONEY; recognises acronyms |
| stem prefixes/suffixes (2–4) | Morphological cues: -oğlu (PERSON), -spor (ORG), -ca/-ce (LANGUAGE) |
| ±2 window of the above | Head words ("Üniversitesi", "Nehri", "Savaşı") decide the type |
| learned label-transition weights | Prevents invalid I- after O and keeps multi-token spans together |

## 4. Deep Contextual Model: BERTurk
`src/train_bert.py`.
- **Encoder.** `dbmdz/bert-base-turkish-cased` with a linear token-classification head.
- **Why this encoder.** It is monolingual Turkish, so it splits Turkish words into fewer subwords than mBERT or XLM-R. It is cased, so the capitalisation signal is kept. At 110M parameters it fits an 8 GB laptop GPU.
- **Label alignment.** Only the first subword of each word carries the label; the other subwords get -100. Predictions are also read from the first subword, so evaluation is exactly word-level and comparable with the CRF.
- **Training.** 3 epochs, AdamW, learning rate 5e-5, linear schedule with 10% warmup, weight decay 0.01, batch size 32, fp16, max length 192. The best epoch is selected by validation loss, and the test set is used once.

## 5. Novel Method: Suffix-Preserving Entity Swap Augmentation
`src/augment.py`.

**Hypothesis.** Errors concentrate on entities unseen in train, which suggests the model memorises surface forms instead of using context.

**Method.**
1. Build a bank of mentions per type from train, with suffixes stripped.
2. For each sentence that contains entities, add one copy in which each span is replaced (with p = 0.7) by a random mention of the same type.
3. If the original last token carried an apostrophe suffix, re-attach it to the new mention after applying **Turkish vowel harmony** (2-way a/e, 4-way ı/i/u/ü) and **consonant assimilation** (d→t after a voiceless consonant, as in Bitlis'te).

This keeps the sentences grammatical, so the model does not learn the shortcut "ungrammatical sentence means a swapped entity".

**Known limitation.** Buffer consonants (y/n) after vowel-final stems, as in Paris'e vs. Ankara'ya, are not handled. The resulting noise is accepted for augmentation.

## 6. Results & Discussion
_(filled after runs — see `reports/results/*.json`)_
