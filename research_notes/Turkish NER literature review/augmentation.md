# Data Augmentation for NER (Mention Replacement / Entity Swap) — notes for the "novel methodology" section

## Q1. Dai & Adel (2020): what they proposed and when it helped

### Takeaway
Dai & Adel (COLING 2020) compared four simple token- and mention-level augmentations (LwTR, SR, MR, SiS). All of them help a lot with tiny training sets: on i2b2 with 50 sentences, LSTM-CRF F1 rose from 17.1 to about 26. The gains shrink as data grows and are roughly zero, sometimes slightly negative, on the full training set. Mention replacement (MR) gave the largest average gain for the recurrent model (+2.9 F1).

### Cited Findings
- **Citation:** Xiang Dai and Heike Adel. 2020. "An Analysis of Simple Data Augmentation for Named Entity Recognition." *Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)*, pp. 3861–3867, Barcelona (online). DOI 10.18653/v1/2020.coling-main.343. arXiv:2010.11683. — [ACL Anthology](https://aclanthology.org/2020.coling-main.343/); [arXiv](https://arxiv.org/abs/2010.11683)
- **Methods** (Sec. 3), each applied independently with a binomial draw of probability p — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2010.11683)
  - *Label-wise token replacement (LwTR):* a token is replaced by another training-set token with the same label. The label sequence stays the same.
  - *Synonym replacement (SR):* a token is replaced by a WordNet synonym. Multi-word synonyms get BIO labels by rule: B becomes B then I…; I becomes all I.
  - *Mention replacement (MR):* a whole entity mention is replaced by another training-set mention **of the same entity type**, and the BIO labels are adjusted to its length. This is the "entity swap".
  - *Shuffle within segments (SiS):* the sentence is split into runs of identical labels, and tokens are shuffled inside a run.
  - *Hyperparameters:* augmented copies per sentence ∈ {1, 3, 6, 10} ({1, 2, 3} when all four are combined); p ∈ {0.1, 0.3, 0.5, 0.7}.
- **Setup:** two domain datasets, i2b2-2010 (biomedical) and MaSciP (materials science). Backbones are an LSTM-CRF ("recurrent") and SciBERT-CRF ("transformer"). Training-set sizes are S = 50, M = 150, L = 500 sentences and F = full. Scores are the mean of 5 seeds. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.11683)
- **Low-resource effect sizes (span micro-F1):**
  - i2b2, LSTM-CRF, S: baseline 17.1 → LwTR 26.7, SR 25.9, MR 25.9, SiS 25.2, All 26.9.
  - i2b2, SciBERT, S: 35.1 → MR 40.1, SR 42.3, All 41.5.
  - MaSciP, LSTM-CRF, S: 53.0 → MR 60.6, All 60.8.
  - MaSciP, SciBERT, S: 68.1 → MR 70.5, All 71.2.
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.11683)
- **Average Δ F1 across sizes and datasets:**
  - Recurrent: LwTR +2.6, SR +2.5, **MR +2.9**, SiS +2.0, All +3.3.
  - Transformer: LwTR +1.0, SR +1.6, MR +1.2, SiS +1.1, All +1.8.
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.11683)
- **Full data:** the gains disappear or reverse.
  - i2b2, SciBERT: baseline 87.8; MR 87.8, LwTR 87.5, All 87.2.
  - MaSciP, LSTM: baseline 76.4; MR 76.0, SiS 77.1.
  - Combining all four methods is not always best. The authors conclude that augmentation is "more effective when the training sets are small."
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.11683)

### Inferences
- MR is the most natural of the four for a Turkish entity-swap method. It keeps the context intact and only changes the name, and its label adjustment is trivial.
- For a full-size Turkish dataset such as WikiANN-tr or the Tür et al. news corpus, expect about ±0.5 F1 from naive MR. Larger gains should be framed as a low-resource or robustness result, not a full-data result.
- These results come from English scientific/biomedical domains. Nobody has tested them on an agglutinative language.

### Gaps
- One claim came through an automated summary of the paper's text and should be checked against the PDF before quoting: that "only three methods achieve significant improvements" on full data.
- The paper does not evaluate on general-domain NER (CoNLL), so it gives no PER/LOC/ORG-specific numbers.

## Q2. Other entity-substitution, generative, and counterfactual methods, and evidence that models rely on names

### Takeaway
Generative augmentation (DAGA) and entity-focused generation (MELM) beat simple replacement in low-resource settings. MELM's key finding is that **diversifying entities is more effective than adding context patterns**. Probing studies (Lin et al. 2020; Agarwal et al. 2020a/b) show that NER models rely heavily on name regularity and memorized mentions. This is the main motivation for entity-swap augmentation: it breaks memorization and forces the model to use context.

### Cited Findings
- **DAGA:** Bosheng Ding, Linlin Liu, Lidong Bing, Canasai Kruengkrai, Thien Hai Nguyen, Shafiq Joty, Luo Si, Chunyan Miao. 2020. "DAGA: Data Augmentation with a Generation Approach for Low-resource Tagging Tasks." *EMNLP 2020*, pp. 6045–6057. DOI 10.18653/v1/2020.emnlp-main.488. — [ACL Anthology](https://aclanthology.org/2020.emnlp-main.488)
  - Method: a language model is trained on *linearized* labeled sentences, with label tokens inserted before words, and then samples new tagged sentences. It needs no external resources such as WordNet, and it can condition on unlabeled data or a knowledge base.
  - Results: it consistently beats baselines on NER, POS tagging and target-based sentiment, with the biggest gains when gold data is scarce.
  - Source: [arXiv 2011.01549](https://arxiv.org/abs/2011.01549)
  - The full author list is from memory and matches the Anthology listing "Ding, Liu, and six co-authors". Verify it on the Anthology page before printing.
- **MELM:** Ran Zhou, Xin Li, Ruidan He, Lidong Bing, Erik Cambria, Luo Si, Chunyan Miao. 2022. "MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER." *ACL 2022 (Long)*, pp. 2251–2262. — [ACL Anthology](https://aclanthology.org/2022.acl-long.160); [arXiv 2108.13655](https://arxiv.org/abs/2108.13655)
  - Method: label tokens are inserted around entity tokens, a pretrained MLM is fine-tuned with **only entity tokens masked**, and new entities are generated in place while the context is kept. This is a learned, context-aware mention replacement that avoids token–label misalignment.
  - Data: CoNLL-2002/2003 (En/De/Es/Nl) with N ∈ {100, 200, 400, 800} sentences.
  - Monolingual average F1, gold-only → MELM: 38.65 → 70.44 (N = 100), 66.52 → 78.27 (200), 76.81 → 81.87 (400), 82.95 → 84.37 (800).
  - MELM vs. DAGA: +9.90 at N = 100, falling to +0.92 at N = 800.
  - Cross-lingual: 38.74 → 57.00 (N = 100).
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2108.13655)
  - Key insight from the preliminary experiment (Fig. 1): "diversifying entities in the training data is more effective than introducing more context patterns." MELM also introduces many more unseen valid entities than baselines. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2108.13655)
- **Lin et al. 2020:** Hongyu Lin, Yaojie Lu, Jialong Tang, Xianpei Han, Le Sun, Zhicheng Wei, Nicholas Jing Yuan. 2020. "A Rigorous Study on Named Entity Recognition: Can Fine-tuning Pretrained Model Lead to the Promised Land?" *EMNLP 2020*, pp. 7291–7300. — [ACL Anthology](https://aclanthology.org/2020.emnlp-main.592); [arXiv 2004.12126](https://arxiv.org/abs/2004.12126)
  - Method: randomization tests. *Name Permutation* removes name regularity, and *Mention Permutation* removes mention coverage.
  - Conclusions:
    1. Name regularity is critical for generalizing to unseen mentions.
    2. High mention coverage can *undermine* generalization because models memorize popular mentions.
    3. Pretrained encoders may need relatively little data to capture context patterns.
  - They also release a Wikipedia-based NER dataset of weak-regularity, low-coverage types (book, song, movie).
  - Source: [ACL Anthology](https://aclanthology.org/2020.emnlp-main.592)
- **Agarwal et al. 2020a, entity-switch probe:** Oshin Agarwal, Yinfei Yang, Byron C. Wallace, Ani Nenkova. "Entity-Switched Datasets: An Approach to Auditing the In-Domain Robustness of Named Entity Recognition Models." arXiv:2004.04123.
  - Method: entities in existing test sets are replaced with names from different countries while the context is kept coherent.
  - Result: in identical contexts, F1 is highest for American and Indian names and lowest for Vietnamese and Indonesian names.
  - Source: [arXiv](https://arxiv.org/abs/2004.04123)
- **Agarwal et al. 2020b:** "Interpretability Analysis for Named Entity Recognition…" (arXiv:2004.04564). Context does inform predictions somewhat, but "the major driver of performance is recognition of certain words as names of a particular type." — [ar5iv](https://ar5iv.labs.arxiv.org/html/2004.04564) (cited via the search summary; the full title was not fetched)
- **RockNER** (arXiv:2109.05620) builds adversarial NER test sets by replacing entities with other same-type entities, plus context perturbations. It describes "name regularity bias" as relying on the name while ignoring the local context. — [arXiv](https://arxiv.org/pdf/2109.05620) (cited only for the concept; not fetched)
- **LLM-DA** (Ye et al. 2024) also uses same-type entity replacement together with LLM context rewriting. See Q4.

### Inferences
- These sources together make a coherent case for a "novel methodology": entity swap acts as both augmentation (more entity diversity, per MELM) and regularization against memorization (Lin et al.).
- Agarwal et al.'s entity-switch protocol can be reused as an **evaluation** probe. For example, swap Turkish PER/LOC names with foreign or rare ones in the test set and measure the F1 drop. This shows robustness and is not just an in-distribution F1 claim.

### Gaps
- I did not find exact F1 drops from Lin et al.'s permutation tests or Agarwal's per-country numbers. Fetch the PDFs if numbers are needed.
- The 2020b interpretability paper's full title and venue were not verified.

## Q3. Morphology-aware augmentation for agglutinative languages (Turkish/Finnish/Hungarian)

### Takeaway
I found **no published method** that performs entity or mention replacement for NER while explicitly re-inflecting the suffixes on the swapped entity (vowel harmony, consonant assimilation) in Turkish, Finnish or Hungarian. The closest work is morphology-aware augmentation for other tasks (Şahin & Steedman 2018, POS tagging) and morphology-aware Turkish NER *models*, not augmentation. This is a real gap and a defensible novelty claim, phrased cautiously as "to our knowledge".

### Cited Findings
- Gözde Gül Şahin and Mark Steedman. 2018. "Data Augmentation via Dependency Tree Morphing for Low-Resource Languages." *EMNLP 2018*, pp. 5004–5009. DOI 10.18653/v1/D18-1545. — [ACL Anthology](https://aclanthology.org/D18-1545/)
  - Method: "crop" (deleting dependency links) and "rotate" (moving tree fragments around the root), applied to low-resource Universal Dependencies treebanks for POS tagging.
  - Result: improvements for most languages, "especially for languages with rich case marking systems." Turkish is one of the languages covered.
- Yıldız, Güngör et al., "Morphological Embeddings for Named Entity Recognition in Morphologically Rich Languages" (arXiv:1706.00506).
  - Method: encodes the morphological analysis of each word for NER, concatenated with word and character embeddings.
  - Result: 93.59% F1 on Turkish.
  - This is morphology on the model side, not augmentation. — [arXiv](https://arxiv.org/pdf/1706.00506)
- Güngör et al., "The Effect of Morphology in Named Entity Recognition with Sequence to Sequence Tagging," *Natural Language Engineering* (2018/2019): morphological tagging schemes for NER in morphologically rich languages, including Turkish. — [PDF](https://www.cmpe.boun.edu.tr/~gungort/papers/The%20Effect%20of%20Morphology%20in%20Named%20Entity%20Recognition%20with%20Sequence%20to%20Sequence%20Tagging,%20Natural%20Language%20Engineering.pdf)
- Oguz Deniz, N. Ceyda Unsoy, Bahaeddin Eravci. 2025. "Large Language Model Based Data Augmentation for Turkish Named Entity Recognition." IEEE conference paper (TOBB ETÜ repository).
  - Motivation: the shortage of annotated Turkish data hurts NER performance.
  - The method and numbers could not be verified, and I could not see whether suffixes are handled. — [TOBB ETÜ repository](https://gcris.etu.edu.tr/handle/20.500.11851/12717)
- Related agglutinative work:
  - A Mongolian NER study found that segmenting case and inflectional suffixes into separate tokens beats deleting them. — [COLING 2016 C16-1049](https://preview.aclanthology.org/fix_video/C16-1049.pdf)
  - A Kazakh NER paper uses an FST morphological analyzer with 3,039 rules. — [NU repository](https://nur.nu.edu.kz/bitstreams/f038ab34-31bd-4a65-a69c-7ecf6cfecdb2/download)
  - Neither does augmentation.

### Inferences
These are my own reasoning, not sourced.
- In Turkish, proper-noun suffixes are written after an apostrophe (e.g., *Ankara'da*, *İzmir'de*, *Paris'te*). Naive MR that copies the whole token `Ankara'da` → `Paris'da` produces ungrammatical forms that violate vowel harmony (a/e) and consonant assimilation (d/t after voiceless consonants).
- Two practical fixes follow:
  - Split at the apostrophe, swap the stem, and regenerate the suffix with harmony rules or a morphological generator such as TRmorph/Zemberek.
  - Swap only mentions that have no suffix.
- The apostrophe convention makes Turkish unusually easy to augment in a morphology-aware way compared with Finnish or Hungarian, which attach case endings without a delimiter and have stem alternations.
- A suffix-correct MR baseline vs. a naive MR ablation would be a clean, cheap experiment for the report.

### Gaps
- No sources were found for Finnish or Hungarian NER augmentation. The search returned nothing language-specific.
- The Deniz et al. (2025) full text and exact venue (IEEE SIU 2025?) were not verified.

## Q4. Gazetteer integration and LLM-based data generation (brief)

### Takeaway
GEMNET (Meng et al., NAACL 2021) is the standard modern citation for gating gazetteer features against context. It reports large gains on low-context or complex entities. For LLM generation, LLM-DA (Ye et al. 2024, arXiv) combines LLM context rewriting with same-type entity replacement for few-shot NER. Turkish has a gazetteer-based auto-annotated corpus.

### Cited Findings
- **GEMNET:** Tao Meng, Anjie Fang, Oleg Rokhlenko, Shervin Malmasi. 2021. "GEMNET: Effective Gated Gazetteer Representations for Recognizing Complex Entities in Low-context Input." *NAACL 2021*, pp. 1499–1512. DOI 10.18653/v1/2021.naacl-main.118. — [ACL Anthology](https://aclanthology.org/2021.naacl-main.118)
  - Method: a contextual gazetteer encoder plus a Mixture-of-Experts gate that weights gazetteer vs. context features. The gate addresses gazetteer over- or under-use.
  - Results: up to **+49% F1 on difficult entities** (uncased), and a new uncased SOTA on CoNLL03 and WNUT17.
  - Data: introduces LOWNER, MSQ-NER and ORCAS-NER (24M tokens).
  - Source: [ACL Anthology](https://aclanthology.org/2021.naacl-main.118)
- **Turkish gazetteer resource:** "Automatically Annotated Turkish Corpus for Named Entity Recognition and Text Categorization using Large-Scale Gazetteers" (arXiv:1702.02363). This is a large Turkish NER corpus auto-labelled with gazetteers. — [arXiv](https://arxiv.org/pdf/1702.02363) (authors and venue not verified in this session)
- **LLM-DA:** Junjie Ye et al. 2024. "LLM-DA: Data Augmentation via Large Language Models for Few-Shot Named Entity Recognition." arXiv:2402.14568.
  - Method: 14 contextual rewriting strategies, same-type entity replacement, and noise injection, followed by filtering and annotation.
  - Results: reports significant few-shot NER gains and higher data quality than prior augmentation methods.
  - Source: [arXiv](https://arxiv.org/abs/2402.14568)
- "Does Synthetic Data Help Low-Resource Named Entity Recognition?" (arXiv:2505.16814, 2025) studies LLM-generated synthetic NER data across 11 low-resource languages. — [arXiv](https://arxiv.org/html/2505.16814v1) (cited only for scope; findings not fetched)

### Inferences
- A gazetteer is the natural *source* of replacement entities for MR. Sampling Turkish PER/LOC/ORG names from a gazetteer, instead of only training-set mentions, increases entity diversity, which is the MELM insight, without needing a generator model.

### Gaps
- No headline numbers were verified for LLM-DA, the 2505.16814 study, or Deniz et al. 2025.
- No LLM-generation study specific to Turkish NER was verified beyond the Deniz et al. abstract.
