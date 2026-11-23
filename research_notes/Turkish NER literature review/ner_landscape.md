# NER Landscape: From Feature-Based Models to LLMs

Verification note: every entry marked [verified] was checked this session against arXiv, the ACL Anthology, ar5iv, or the ACL wiki. Entries marked [unverified] come from background knowledge and should be checked before they are cited.

## Q1. What CoNLL-2003 English F1 does each paradigm reach (approximately)?

### Takeaway
English CoNLL-2003 test F1 rose from about 88.8 for feature-based classifier ensembles (2003), to about 91 for BiLSTM-CRF (2016), to about 93 for contextual string embeddings and fine-tuned BERT or XLM-R (2018-2020), and to about 94.3-94.6 for entity-aware or document-context encoders (LUKE, ACE+doc). Gains slowed sharply after 2018: the jump from Flair to LUKE was about 1.2 points.

### Cited Findings (chronological)
- **Rule and gazetteer era (MUC).** [unverified] MUC-6 and MUC-7 (Grishman & Sundheim 1996, "Message Understanding Conference-6: A Brief History", COLING) introduced the NER task. Handcrafted rule and gazetteer systems led those evaluations. No CoNLL-2003 number exists for this era.
- **CRF.** [unverified, canonical] Lafferty, McCallum & Pereira (2001), "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data", ICML 2001. This is the feature-based sequence model that dominated NER until about 2015. Augenstein et al. (2017) report that the CRF systems Stanford NER and CRFSuite reach about 91-95 F1 on *seen* CoNLL TestB entities but only about 81-83 F1 on unseen ones (see Q3) — [ar5iv 1701.02877](https://ar5iv.arxiv.org/html/1701.02877)
- **CoNLL-2003 shared task.** [verified] Tjong Kim Sang & De Meulder (2003), "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition", CoNLL 2003 (W03-0419). The best English system was Florian, Ittycheriah, Jing & Zhang (2003), "Named Entity Recognition through Classifier Combination", with **88.76 F1**. The vocabulary-transfer baseline scored 59.61 F1 — [ACL wiki CoNLL-2003 SotA](https://aclweb.org/aclwiki/CONLL-2003_(State_of_the_art)); [W03-0419](https://aclanthology.org/W03-0419.pdf)
- **BiLSTM-CRF (Huang et al.).** [verified via wiki] Huang et al. (2015), BI-LSTM-CRF, **90.10 F1** — [ACL wiki](https://aclweb.org/aclwiki/CONLL-2003_(State_of_the_art))
- **BiLSTM-CRF (Lample et al.).** [verified] Lample, Ballesteros, Subramanian, Kawakami & Dyer (2016), "Neural Architectures for Named Entity Recognition", NAACL-HLT 2016, pp. 260-270. The LSTM-CRF with pretrained embeddings, dropout and character features reaches **90.94 F1**. Without pretrained embeddings it drops to 83.63, and without dropout to 88.39. It uses no gazetteers or hand-crafted features — [ar5iv 1603.01360](https://ar5iv.labs.arxiv.org/html/1603.01360)
  - Conneau et al. (2020) reproduce Lample at 90.74 in their table. Small discrepancies like this are normal — [ar5iv 1911.02116](https://ar5iv.labs.arxiv.org/html/1911.02116)
- **BiLSTM-CNN-CRF.** [verified] Ma & Hovy (2016), "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF", ACL 2016: **91.21 F1** — [ar5iv 1603.01354](https://ar5iv.labs.arxiv.org/html/1603.01354)
- **Contextual string embeddings (Flair).** [verified] Akbik, Blythe & Vollgraf (2018), "Contextual String Embeddings for Sequence Labeling", COLING 2018 (C18-1139): **93.09 ± 0.12 F1**. The previous best was 92.22 (Peters et al. 2018, ELMo) — [ACL Anthology C18-1139](https://aclanthology.org/C18-1139); [Papers with Code](https://paperswithcode.com/paper/contextual-string-embeddings-for-sequence)
- **BERT.** [verified] Devlin, Chang, Lee & Toutanova (2019), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019. With fine-tuning, BERT-Large scores **92.8 test** (96.6 dev) and BERT-Base 92.4 test (96.4 dev). The best feature-based variant (concatenating the last four hidden layers) scores 96.1 dev — [ar5iv 1810.04805](https://ar5iv.labs.arxiv.org/html/1810.04805)
- **XLM-R.** [verified] Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer & Stoyanov (2020), "Unsupervised Cross-lingual Representation Learning at Scale", ACL 2020 (2020.acl-main.747). English F1: XLM-R (large) **92.92** and XLM-R Base 92.25, against mBERT 91.97 (from Wu & Dredze 2019) and Akbik et al. 93.18. Training on all languages gives 92.00. XLM-R uses no CRF, which the authors say gives Akbik et al. an advantage. This is relevant for Turkish because XLM-R is the standard multilingual encoder baseline — [ar5iv 1911.02116](https://ar5iv.labs.arxiv.org/html/1911.02116); [ACL Anthology](https://aclanthology.org/2020.acl-main.747)
- **LUKE.** [verified] Yamada, Asai, Shindo, Takeda & Matsumoto (2020), "LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention", EMNLP 2020 (2020.emnlp-main.523). It scores **94.3 F1**, against 92.4 for a RoBERTa trained on the same Wikipedia corpus. Entity-aware attention reaches 94.3, versus 94.1 with standard attention. LUKE-base scores 93.3 — [ACL Anthology](https://aclanthology.org/2020.emnlp-main.523); [GitHub studio-ousia/luke](https://www.github.com/studio-ousia/luke)
- **ACE + document context.** [verified, as a baseline cited by others] This is the supervised SOTA that GPT-NER and Han et al. compare against, at **94.6 F1** — [ar5iv 2304.10428](https://ar5iv.labs.arxiv.org/html/2304.10428); [ar5iv 2305.14450](https://ar5iv.labs.arxiv.org/html/2305.14450)

### Summary table (CoNLL-2003 English test F1)
| Paradigm | Representative | F1 |
|---|---|---|
| Feature-based ensemble (2003) | Florian et al. 2003 | 88.76 |
| BiLSTM-CRF | Lample et al. 2016 / Ma & Hovy 2016 | 90.94 / 91.21 |
| Contextual char-LM embeddings | Akbik et al. 2018 (Flair) | 93.09 |
| Fine-tuned Transformer encoder | BERT-Large 2019 / XLM-R 2020 | 92.8 / 92.92 |
| Entity-aware encoder | LUKE 2020 | 94.3 |
| Supervised SOTA (doc context) | ACE + doc-context | 94.6 |
| LLM zero-shot (no fine-tuning) | GPT-4 / GPT-3.5 (Han et al.) | 72.3 / 60.1 |
| LLM few-shot ICL | GPT-4 / GPT-3.5 (Han et al.) | 78.5 / 70.5 |
| LLM + kNN retrieval + self-verification | GPT-NER (GPT-3 davinci-003), full test | 90.91 |

### Inferences
- The largest jumps came from learned features (BiLSTM-CRF, about +2 over 2003) and from contextual pretraining (Flair or BERT, about +2 more). Since 2018 the ceiling has risen only about 1.5 points, partly because about 5% of test sentences contain label errors (see Gaps and Q3). That makes differences under 0.5 F1 on CoNLL-2003 hard to interpret.
- BERT-Large (92.8) and Flair (93.09) are statistically close. The most fair one-line characterisation is "pretrained contextual encoders ≈ 93".

### Gaps
- I did not verify the exact MUC-era citation or Lafferty et al. (2001) this session (searches were capped at 12). Both are canonical, but check the bibliographic details before citing them.
- Survey to cite [unverified]: Li, Sun, Han & Li (2022), "A Survey on Deep Learning for Named Entity Recognition", IEEE TKDE.

## Q2. What evidence is there that zero-/few-shot LLMs underperform fine-tuned encoders on NER?

### Takeaway
In-context LLMs fall well short of fine-tuned encoders on NER. On CoNLL-2003, GPT-4 scores 72.3 F1 zero-shot and 78.5 few-shot, against a supervised SOTA of 94.6. The gap is much larger for nested NER. LLM approaches catch up only when they borrow supervision: GPT-NER uses a fine-tuned NER model to retrieve demonstrations, and UniversalNER is fine-tuned (distilled). Even then, a supervised UniNER-7B (93.3) only matches encoders such as LUKE (94.3) on CoNLL-2003.

### Cited Findings
- **Han et al. (2023/2024).** [verified] Han, Yang, Peng, Tiwari, Wan, Liu & Wang, "Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors", arXiv 2305.14450. The extended version, "An Empirical Study on Information Extraction using Large Language Models", appeared in *Neural Networks*. The study covers 17 datasets and 14 IE subtasks and finds "a huge performance gap between ChatGPT and SOTA results", with harder tasks showing larger gaps — [arXiv 2305.14450](https://arxiv.org/abs/2305.14450v1); [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.14450)
  - NER F1 for GPT-4, with GPT-3.5-turbo in parentheses, against SOTA:
    - CoNLL03: 72.30 (60.10) zero-shot, 78.50 (70.53) few-shot ICL, 76.19 (74.63) few-shot CoT. SOTA is 94.6.
    - FewNERD: 47.84 (31.56) zero-shot. SOTA is 67.1.
    - Nested NER, zero-shot: ACE04 31.43 (27.80) vs SOTA 88.5; ACE05 24.68 (23.38) vs 87.5; GENIA 46.22 (38.09) vs 81.5.
    - Source: [ar5iv 2305.14450](https://ar5iv.labs.arxiv.org/html/2305.14450)
  - Caveat: the authors argue that exact span matching penalises LLMs and propose soft matching. They also report that the most common ChatGPT error is "unannotated spans", which may reflect gaps in the gold annotation — [arXiv 2305.14450](https://arxiv.org/abs/2305.14450v1)
- **Qin et al. (2023).** [verified] Qin, Zhang, Zhang, Chen, Yasunaga & Yang, "Is ChatGPT a General-Purpose Natural Language Processing Task Solver?", EMNLP 2023, pp. 1339-1384. The study covers 20 datasets and 7 task categories, zero-shot. ChatGPT "still struggles with specific tasks such as sequence tagging", including NER — [ACL Anthology 2023.emnlp-main.85](https://aclanthology.org/2023.emnlp-main.85). (I did not retrieve its exact CoNLL03 numbers.)
- **GPT-NER (Wang et al. 2023).** [verified] Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li & Guoyin Wang, "GPT-NER: Named Entity Recognition via Large Language Models", arXiv 2304.10428 (2023), published in Findings of NAACL 2025 (2025.findings-naacl.239).
  - Method: NER is recast as generation that marks entities as `@@entity##`, with one prompt per entity type. Demonstrations are retrieved by kNN over entity-level embeddings, and a self-verification step re-asks the LLM whether each extracted span belongs to the type.
  - Full CoNLL-2003 test: **90.91 F1** with GPT-3 (davinci-003). That is below BERT-Tagger 92.8, BERT-MRC 93.04, GNN-SL 93.2 and ACE+doc 94.6.
  - Retrieval matters a lot: random demonstrations score 73.18, sentence-level kNN 84.36, entity-level kNN 89.97.
  - Important caveat: the entity-level kNN embeddings come from a *fine-tuned NER model*, so the best setting is not purely zero-shot.
  - Many reported figures (95.85 F1) are on a **100-sentence subset**, where SOTA is 98.04. Do not compare these with full-test numbers.
  - Low-resource strength: with only 8 training sentences GPT-NER scores about 60 F1, while supervised models score near 0. Supervised models overtake it as data grows.
  - Sources: [ar5iv 2304.10428](https://ar5iv.labs.arxiv.org/html/2304.10428); [arXiv](https://arxiv.org/abs/2304.10428v4); [NAACL Findings 2025](https://preview.aclanthology.org/watermark/2025.findings-naacl.239)
- **UniversalNER (Zhou et al. 2024).** [verified] Zhou, Zhang, Gu, Chen & Poon, "UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition", ICLR 2024 (arXiv 2308.03279).
  - Method: ChatGPT is distilled into LLaMA-based students through NER-focused instruction tuning. The paper also builds a 43-dataset, 9-domain benchmark — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/34678d08b36076de986df95c5bbba92f-Abstract-Conference.html)
  - Zero-shot average F1: UniNER-7B 41.7, UniNER-13B 43.4, ChatGPT 34.9. UniNER beats Vicuna and Alpaca by more than 30 points, and Vicuna trails ChatGPT by more than 20 — [ar5iv 2308.03279](https://ar5iv.labs.arxiv.org/html/2308.03279)
  - Supervised, in-domain over 20 datasets: UniNER-7B averages 84.78, InstructUIE-11B 81.16, BERT-base 80.09. On CoNLL-2003 specifically: UniNER-7B 93.30, BERT-base 92.40, InstructUIE 91.53 — [ar5iv 2308.03279](https://ar5iv.labs.arxiv.org/html/2308.03279)
  - The paper's text and its table disagree on one out-of-domain number (60.0 vs 61.8) — [ar5iv 2308.03279](https://ar5iv.labs.arxiv.org/html/2308.03279)

### Inferences
- Zero-shot open-domain NER averages around 35-43 F1 for both UniNER and ChatGPT. In-context LLMs are useful when labelled data is scarce, as GPT-NER shows in its 8-sentence setting. But with a few thousand labelled sentences, a fine-tuned encoder of about 100-500M parameters is both cheaper and more accurate.
- For a Turkish case study the gap is likely larger, because Han et al. find that long-tail types hurt LLMs and the English-centric LLMs above were not evaluated on Turkish. This is an inference, not a measured result.

### Gaps
- Exact zero-shot CoNLL03 F1 from Qin et al. (2023) was not retrieved.
- [unverified] Xie et al. (2023), "Empirical Study of Zero-Shot NER with ChatGPT" (EMNLP 2023), is another candidate citation I did not check.
- I found no Turkish-specific evaluation of LLMs versus fine-tuned encoders in this session.

## Q3. Which paper(s) show NER models memorise seen entities and fail on unseen ones? (plus other challenges)

### Takeaway
Augenstein, Derczynski & Bontcheva (2017) is the key citation. On CoNLL, about 50% of TestB entities are unseen in training, and every system scores lower on them, mainly because recall drops. The paper concludes that "memorisation consistently predicts NER performance". Taillé et al. (2020) and Fu et al. (2020) extend the analysis to contextual embeddings and neural models. The main remaining challenges are label noise (about 5.4% of CoNLL-2003 test sentences contain errors), nested entities (where LLMs collapse to 24-46 F1), and domain and temporal shift.

### Cited Findings: memorisation and unseen entities
- **Augenstein et al. (2017).** [verified] Augenstein, Derczynski & Bontcheva, "Generalisation in Named Entity Recognition: A Quantitative Analysis", *Computer Speech & Language* 44 (2017), pp. 61-83 (arXiv 1701.02877) — [White Rose eprint](https://eprints.whiterose.ac.uk/110909/); [arXiv](https://arxiv.org/abs/1701.02877)
  - The abstract says NER approaches "struggle to generalise in diverse genres with limited training data". Unseen entities are more frequent in social media than in newswire, and leading systems rely heavily on surface forms seen in training — [arXiv](https://arxiv.org/abs/1701.02877)
  - Unseen-entity rate in CoNLL: 35.1% of TestA and 49.6% of TestB — [ar5iv](https://ar5iv.arxiv.org/html/1701.02877)
  - Seen vs unseen F1 on CoNLL TestB:

    | System | Seen F1 | Unseen F1 |
    |---|---|---|
    | Stanford NER (CRF) | 95.06 | 82.72 |
    | CRFSuite | 91.80 | 80.99 |
    | SENNA (embeddings) | 93.02 | 89.46 |

    Source: [ar5iv](https://ar5iv.arxiv.org/html/1701.02877)
  - Across all corpora (macro average), seen vs unseen F1 is: Stanford 83.79 vs 48.48, CRFSuite 75.02 vs 45.61, SENNA 79.85 vs 58.08. On ACE newswire, Stanford NER drops from 80.84 to 34.88 — [ar5iv](https://ar5iv.arxiv.org/html/1701.02877)
  - Stanford NER and CRFSuite have about half the recall on unseen entities. SENNA has the smallest gap, which the authors attribute to its embeddings reducing feature sparsity — [ar5iv](https://ar5iv.arxiv.org/html/1701.02877)
- **Follow-up work.** [titles verified via search results only]
  - Taillé, Guigue & Gallinari (2020), "Contextualized Embeddings in Named-Entity Recognition: An Empirical Study on Generalization" (ECIR 2020) — [arXiv 2001.08053](https://arxiv.org/pdf/2001.08053)
  - Fu, Liu, Neubig et al. (2020), "Rethinking Generalization of Neural Models: A Named Entity Recognition Case Study" (AAAI 2020) — [arXiv 2001.03844](https://arxiv.org/pdf/2001.03844)
  - These extend the seen/unseen analysis to BERT- and ELMo-era models. I did not retrieve their numbers. Author lists and venues are from memory.

### Cited Findings: label noise
- **CrossWeigh.** [verified] Zihan Wang, Shang, Liu, Lu, Liu & Han (2019), "CrossWeigh: Training Named Entity Tagger from Imperfect Annotations", EMNLP-IJCNLP 2019 (D19-1519) — [ACL Anthology D19-1519](https://aclanthology.org/D19-1519/)
  - Label mistakes affect about **5.38% of CoNLL-2003 test sentences** (186 sentences), which is significant when SOTA is about 93 F1. Five expert annotators corrected them, and the result is released as "CoNLL++" — [ACL Anthology D19-1519](https://aclanthology.org/D19-1519/); [arXiv 1909.01441](https://arxiv.org/pdf/1909.01441)
  - A later label-error study (arXiv 2210.03920) found these corrections reliable — [arXiv 2210.03920](https://arxiv.org/pdf/2210.03920)
  - "CoNLL#" (arXiv 2405.11865, 2024) offers a further corrected CoNLL-03 English test set with fine-grained error analysis — [arXiv 2405.11865](https://arxiv.org/pdf/2405.11865)

### Cited Findings: temporal and domain shift
- **Liu & Ritter (2023).** [verified] Shuheng Liu & Alan Ritter, "Do CoNLL-2003 Named Entity Taggers Still Work Well in 2023?", ACL 2023, pp. 8254-8271. It won the conference's Reproduction Award — [ACL Anthology 2023.acl-long.459](https://aclanthology.org/2023.acl-long.459)
  - The authors annotate a new test set of 2020 news, confusingly *also* named CoNLL++ (different from Wang et al. 2019), and test more than 20 models.
  - Pretrained Transformers such as RoBERTa and T5 show no clear degradation. Most of the observed drop comes from *temporal mismatch between pre-training data and test data*, not from adaptive overfitting to the test set.
  - Architecture, model size, pre-training data period and amount of fine-tuning data all affect generalisation.
  - Source: [ACL Anthology 2023.acl-long.459](https://aclanthology.org/2023.acl-long.459); [arXiv 2212.09747](https://arxiv.org/abs/2212.09747v2)
- **Genre shift.** Augenstein et al. (2017) show that diverse genres, especially social media, score "significantly lower F1" than newswire — [arXiv 1701.02877](https://arxiv.org/abs/1701.02877)

### Cited Findings: nested NER
- Nested-NER benchmarks (ACE04, ACE05, GENIA) have supervised SOTA of 81.5-88.5 F1. GPT-4 scores only 24.7-46.2 zero-shot and 43.4-57.0 few-shot on them, so nesting is where LLMs lag most — [ar5iv 2305.14450](https://ar5iv.labs.arxiv.org/html/2305.14450)

### Inferences
- With about half of CoNLL TestB entities unseen, and seen/unseen gaps of 3-13 F1 even on newswire, it is good practice to report seen/unseen splits. This matters for a Turkish dataset too, where agglutinative morphology multiplies surface forms and so plausibly raises the unseen rate. That last point is an inference that needs checking against the Turkish data.
- Embedding-based models (SENNA, and later contextual encoders) narrow the seen/unseen gap. This matches the gains seen at each paradigm shift in Q1.

### Gaps
- [unverified, from memory] Finkel & Manning (2009), "Nested Named Entity Recognition", EMNLP, is the classic nested-NER reference. I did not check it.
- [unverified, from memory] For domain shift, Liu et al. (2021), "CrossNER: Evaluating Cross-Domain Named Entity Recognition", AAAI 2021, is the standard benchmark. I did not check it.
- I retrieved no numbers on seen vs unseen performance for BERT, XLM-R or LLMs. Taillé et al. (2020) is the place to get them.
