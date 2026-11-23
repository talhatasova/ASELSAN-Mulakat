# Turkish Named Entity Recognition: Literature Review Notes

## Q1. The turkish-nlp-suite/turkish-wikiNER dataset and its paper (Altinok, ACL 2023)

### Takeaway
Turkish WikiNER is a manually re-annotated (crowd-sourced, by 25 native-speaker annotators at the vendor Co-one) 20,000-sentence subset of the automatically annotated (silver) TWNERTC corpus. It uses 19 OntoNotes-style entity types, not 18. It was released by Duygu Altinok in "A Diverse Set of Freely Available Linguistic Resources for Turkish" (ACL 2023, Long Papers) under CC BY-SA 4.0.

### Cited Findings
- **Citation:** Duygu Altinok. 2023. "A Diverse Set of Freely Available Linguistic Resources for Turkish." In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 13739–13750, Toronto, Canada. DOI 10.18653/v1/2023.acl-long.768. The author's affiliation is Deepgram, and the paper was a poster in the Resources & Evaluation track. — [ACL Anthology](https://aclanthology.org/2023.acl-long.768); [dblp](https://dblp1.uni-trier.de/rec/conf/acl/Altinok23.html)
- **Origin and annotation:** The paper says: "it is a re-annotation of a subset of the TWNERTC dataset (Sahin et al., 2017), which is a collection of automatically categorized and annotated sentences from Turkish and English Wikipedia." It also says that "Since TWNERTC is an automatically annotated dataset, its label accuracy is not sufficient … we manually annotated a set of 20,000 sentences from TWNERTC." — [Altinok 2023, §3.1.1](https://aclanthology.org/2023.acl-long.768.pdf)
- **Annotators:** "The dataset was annotated by crowd-sourcing; the labeling work was done by a total of 25 annotators (15 female, 10 male). All annotators were native speakers of Turkish residing in Turkey." The data labeling vendor was Co-one, Istanbul. — [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf); [HF dataset card](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
- **Source corpus:** H. Bahadir Sahin, Caglar Tirkaz, Eray Yildiz, Mustafa Tolga Eren, and Ozan Sonmez. 2017. "Automatically Annotated Turkish Corpus for Named Entity Recognition and Text Categorization using Large-Scale Gazetteers." Version 1 of TWNERTC has 4 coarse labels, and version 2 has more than 1,000 fine-grained labels. — [Altinok 2023 (reference list and §3.1.1)](https://aclanthology.org/2023.acl-long.768.pdf)
- **The HF card's provenance statement differs from the paper:** the card says the sentences were "re-annotated from the Kuzgunlar NER dataset," while the paper says TWNERTC. The paper is the primary source. — [HF card](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
- **Label set, 19 types:** CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, TITLE, WORK_OF_ART. The tagging is IOB, with 39 tags (19×B/I + O). The paper's text also says "the following 19 types of entities." The card's metadata says 18, which contradicts its own list. Compared with OntoNotes 5.0's 18 types, TITLE is added. — [HF card](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER); [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf)
- **Label counts (paper Table 1, as extracted):** CARDINAL 4,295; DATE 6,923; EVENT 2,392; FAC 944; GPE 10,368; LANGUAGE 822; LAW 80; LOC 1,364; MONEY 100; NORP 4,023; ORDINAL 1,711; ORG 4,583; PERCENT 182; PRODUCT 12,787; QUANTITY 990; TIME 131; TITLE 2,494; WORK_OF_ART 2,951. The PERSON row was lost in text extraction, but by subtraction from the 57,749 total it is about 3,589 (my inference, not stated). — [Altinok 2023, Table 1](https://aclanthology.org/2023.acl-long.768.pdf)
- **Size:** "20,000 annotated sentences, around 357K words, with 70K of these words being unique; a total of 57,749 named-entities are labeled, and 101K words are labeled as entities." The HF splits are 18,000 train, 1,000 validation, and 1,000 test, and the metadata reports 19,967 rows. — [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf); [HF card](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
- **License:** CC BY-SA 4.0. — [HF card](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
- **Baseline on the HF card (BERTurk fine-tuned):** micro-F1 0.76, macro-F1 0.54, weighted-F1 0.75. PERSON is best at about 0.91. LAW, MONEY, PERCENT, and TIME score 0.00 because of very low support. — [HF card](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)
- **spaCy pipelines (paper Table 5):** NER-F is 0.89 for tr_core_news_md, 0.89 for tr_core_news_lg, and 0.91 for tr_core_news_trf (built on dbmdz BERTurk). The NER component was trained on Turkish Wiki NER plus PanX/WikiANN (Pan et al., 2017), so these numbers are not a pure Wiki-NER test score. — [Altinok 2023, §4, Table 5](https://aclanthology.org/2023.acl-long.768.pdf)
- **Motivation:** Altinok describes the existing Turkish NER data as scarce, with "difficulty of access, lack of commercial usability, small size, and minimal annotation." The Tür et al. data is available by email, the ITU datasets come under a non-commercial licence, and the availability of the 9,358-tweet set from Eken & Tantuğ (2015) is unclear. — [Altinok 2023, §2.1](https://aclanthology.org/2023.acl-long.768.pdf)

### Inferences
- The card's macro-F1 of 0.54 against micro-F1 of 0.76 shows a heavy long tail. Rare types (LAW n=80, MONEY 100, TIME 131, PERCENT 182) dominate macro error. This is a useful framing for any per-class analysis in the report.
- "Silver standard" is the wrong description for Turkish WikiNER itself. It is gold, manual crowd annotation over sentences that originally came from a silver corpus (TWNERTC).

### Gaps
- Inter-annotator agreement is not reported in the paper. The checklist says the guideline is longer than 5 pages and was not included.
- The exact training setup behind the HF card's BERTurk scores (epochs, seed) is not documented on the card.

## Q2. Classic Turkish NER resources and feature-based methods

### Takeaway
Classic Turkish NER is anchored on the Tür, Hakkani-Tür & Oflazer (2003) news corpus of about 500K words with ENAMEX labels (PER/LOC/ORG). On that corpus an HMM-style statistical model reached 91.56% MUC F, and the Şeker & Eryiğit (2012) CRF with morphological and gazetteer features reached 94.59% MUC and 91.94% CoNLL F. On tweets, performance collapses: Küçük et al. (2014) report 42.68% F with an adapted rule-based system, down from 69.12% on news.

### Cited Findings
- **Tür et al. 2003:** Gökhan Tür, Dilek Hakkani-Tür, and Kemal Oflazer. 2003. "A statistical information extraction system for Turkish." *Natural Language Engineering* 9(2):181–210. — [dblp](https://dblp.org/rec/journals/nle/TurHO03); [Cambridge](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/statistical-information-extraction-system-for-turkish/7C288FAFC71D5F0763C1F8CE66464017)
  - The data is newspaper text. Training used 492,821 words with 16,335 person, 11,743 location, and 9,199 organization names (37,277 names in total). The test set is about 28,000 words with 2,197 names (924 PER, 696 LOC, 577 ORG). — [Tür et al. 2003, §4.2.1 (PDF)](https://www.andrew.cmu.edu/user/ko/downloads/Papers/InformationExtraction.pdf)
  - The method is an HMM-based statistical tagger that combines four information sources: lexical (surface forms), contextual cues, morphological analyses, and a tag-sequence model. The paper reports "an F-Measure of 91.56%, according to the MUC evaluation criteria." — [Tür et al. 2003, abstract](https://www.andrew.cmu.edu/user/ko/downloads/Papers/InformationExtraction.pdf)
  - **Counts differ between papers:** Şeker & Eryiğit (2012) describe the same data as "500K words … 24,101 person names, 16,105 location names and 13,540 organization names" with a 1/10 test split (47,344 words), following Yeniterzi (2011). Altinok (2023) cites "37,189 named entities (16,291 person, 11,715 location, 9,183 organization)." — [Şeker & Eryiğit 2012](https://aclanthology.org/C12-1150/); [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf)
- **Şeker & Eryiğit 2012:** Gökhan Akın Şeker and Gülşen Eryiğit. 2012. "Initial Explorations on using CRFs for Turkish Named Entity Recognition." In *Proceedings of COLING 2012*, pp. 2459–2474, Mumbai. ACL Anthology C12-1150. — [ACL Anthology](https://aclanthology.org/C12-1150/)
  - The method is a CRF. Features include surface form or stem, POS, noun-case, proper-noun flag, inflectional features from a two-level morphological analyzer (Oflazer 1994) plus a disambiguator (Sak et al.), case and capitalization features, start-of-sentence, base gazetteers (first and last names, locations from the postal-code system), and "generator" gazetteers (for example *bakanlık* "ministry"). Proper nouns are split at the apostrophe into two tokens. — [Şeker & Eryiğit 2012 PDF](https://aclanthology.org/C12-1150.pdf)
  - **Results (Table 4, CoNLL F):** the final model (+GG) scores PER 92.94, ORG 88.77, LOC 92.93, overall 91.94. The baseline BM_surf scored 82.28 overall. The capitalization feature (+CS) gave the largest single jump (86.59 to 89.59). **Best MUC F is 94.59.** The abstract rounds these to "95% in MUC and 92% in CoNLL." — [Şeker & Eryiğit 2012, Tables 3–5](https://aclanthology.org/C12-1150.pdf)
  - **Prior results compiled in their Table 5:** Özkaya & Diri (2011) 84.24 on e-mail; Küçük & Yazıcı (2012) 90.13 on news; Tür et al. (2003) 91.56 MUC on news; Bayraktar & Temizel (2008) 81.97 MUC on financial text, person names only; Tatar & Cicekli (2011) 91.08 CoNLL on terrorism news; Yeniterzi (2011) 88.94 CoNLL on news (CRF with IG-based tokenization). — [Şeker & Eryiğit 2012](https://aclanthology.org/C12-1150.pdf)
- **ITU NER datasets:** Altinok notes the ITU NLP group "offers three NER datasets (Şeker and Eryiğit, 2017)" with PER/ORG/LOC, under a licence that forbids commercial use. — [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf)
- **Küçük et al. 2014 (tweets):** Dilek Küçük, Guillaume Jacquet, and Ralf Steinberger. 2014. "Named Entity Recognition on Turkish Tweets." In *Proceedings of LREC 2014*, pp. 450–454, Reykjavik. ACL Anthology L14-1328. — [ACL Anthology](https://aclanthology.org/L14-1328/)
  - The corpus has 2,320 tweets from 26 July 2013, 20,752 words, and 1,322 NEs, of which 980 are PLOs (person, location, organization). It is distributed by the European Commission's Joint Research Centre (JRC). — [Küçük et al. 2014 PDF](https://aclanthology.org/L14-1328.pdf)
  - The method is a rule- and gazetteer-based multilingual system, the JRC's Europe Media Monitor (EMM) NER system, adapted to Turkish. With the modified system, overall PLO F is **69.12%** on news (the METU-corpus set, about 20K tokens), **42.68%** on their tweet set, and **36.11%** on the Çelikkaya et al. (2013) tweet set (about 54K tokens, 1,374 PLOs). A prior rule-based system (Küçük & Yazıcı 2009) reached 78.7% on the same news set. — [Küçük et al. 2014](https://aclanthology.org/L14-1328.pdf)
  - It cites Çelikkaya et al. (2013): Şeker & Eryiğit's news-trained CRF dropped "from 91% to 19%" F when applied to tweets. — [Küçük et al. 2014](https://aclanthology.org/L14-1328.pdf)
  - Follow-up: Dilek Küçük and Ralf Steinberger. 2014. "Experiments to Improve Named Entity Recognition on Turkish Tweets." In *Proceedings of the 5th Workshop on Language Analysis for Social Media (LASM)*, pp. 71–78, Gothenburg. — [ACL Anthology W14-1309](https://aclanthology.org/W14-1309/)
- **Other tweet data:** Eken & Tantuğ (2015) have a 9,358-tweet NER set of unclear availability. — [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf)

### Inferences
- MUC and CoNLL scores are not comparable. MUC gives partial credit and runs about 2–3 points higher here, so a report should label the metric whenever it quotes pre-neural numbers.
- The pre-neural feature hierarchy is consistent across papers: capitalization, then morphology (stems and inflections), then gazetteers. Capitalization and apostrophes are exactly the cues that vanish in tweets.

### Gaps
- I did not open the LASM 2014 follow-up's numbers or the Şeker & Eryiğit 2017 dataset paper, so their exact F1 and citation details are unverified.
- The Tür 2003 per-model breakdown (Tables 17–18) was not extracted beyond the final 91.56%.

## Q3. Neural Turkish NER: morphology-aware models, BERTurk, XLM-R and mBERT

### Takeaway
Morphology-aware BiLSTM-CRF models (Güngör et al.) reached 93.59% F1 on the Tür et al. news data. BERTurk (Schweter 2020) reaches about 93.9 test F1 when fine-tuned in-language on the Turkish WikiANN split. In XTREME's zero-shot setting (fine-tuned on English only), Turkish WikiANN F1 is 71.8 for mBERT and 76.1 for XLM-R (large). Transformer+CRF models were later reported at 95.95% F on the news benchmark.

### Cited Findings
- **Güngör et al. 2017 (preprint):** Onur Güngör, Eray Yıldız, Suzan Üsküdarlı, and Tunga Güngör. 2017. "Morphological Embeddings for Named Entity Recognition in Morphologically Rich Languages." arXiv:1706.00506 (marked as a working draft). It extends the Lample et al. (2016) BiLSTM-CRF with morphological embeddings, built by a Bi-LSTM over the tags of each word's morphological analysis and combined with word and character embeddings. The paper reports "new state-of-the-art results of 93.59% and 79.59%" F1 for Turkish and Czech. — [arXiv](https://arxiv.org/abs/1706.00506)
- **Güngör et al. 2019 (journal version):** Onur Güngör, Tunga Güngör, and Suzan Üsküdarlı. 2019. "The effect of morphology in named entity recognition with sequence tagging." *Natural Language Engineering* 25(1):147–169. DOI 10.1017/S1351324918000281. It reports state-of-the-art results for Turkish, Czech, Hungarian, Finnish, and Spanish. — [author PDF](https://www.cmpe.boun.edu.tr/~gungort/papers/The%20Effect%20of%20Morphology%20in%20Named%20Entity%20Recognition%20with%20Sequence%20to%20Sequence%20Tagging,%20Natural%20Language%20Engineering.pdf); [BL catalog](https://eld.bl.uk/catalog/100077764085_0x000001)
- **Joint learning:** a related 2018 preprint, "Improving Named Entity Recognition by Jointly Learning to Disambiguate Morphological Tags" (arXiv:1807.06683, Güngör et al.), jointly learns morphological disambiguation and NER. — [arXiv](https://arxiv.org/abs/1807.06683)
- **Aras et al. 2020:** Gizem Aras, Didem Makaroğlu, Şeniz Demir, and Altan Çakır. 2020. "An Evaluation of Recent Neural Sequence Tagging Models in Turkish Named Entity Recognition." arXiv:2005.07692 (submitted to *Expert Systems with Applications*). A transformer + CRF model reaches **95.95% F** on "a common dataset," reported as state of the art. — [arXiv](https://arxiv.org/abs/2005.07692)
- **BERTurk citation:** Stefan Schweter. 2020. "BERTurk – BERT models for Turkish." Zenodo, v1.0.0, April 2020. DOI 10.5281/zenodo.3770924. The HF model is dbmdz/bert-base-turkish-cased. A v2, "BERTurk v2" (March 2025, DOI 10.5281/zenodo.14963493), covers newer models such as BERT5urk. — [stefan-it/turkish-bert](https://github.com/stefan-it/turkish-bert)
- **BERTurk NER results** (Turkish WikiANN split, 10 epochs, F1 averaged over 5 seeds, test F1):
  - BERTurk cased 128k vocabulary: **93.92 ± 0.16** (dev 93.92)
  - BERTurk uncased 128k: 93.29
  - BERTurk cased 32k: 93.26
  - BERTurk uncased 32k: 92.96
  - ConvBERTurk cased: 93.93
  - ELECTRA Base cased: 93.60
  - DistilBERTurk: 91.17
  - BERT5urk: 93.66

  The repo README does not report mBERT or XLM-R numbers. — [stefan-it/turkish-bert](https://github.com/stefan-it/turkish-bert)
- **XTREME:** Hu, Ruder, Siddhant, Neubig, Firat, and Johnson. 2020. "XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization." arXiv:2003.11080 (ICML 2020). Table 21 gives zero-shot NER F1 on WikiANN/PAN-X, trained on English and tested on Turkish:
  - mBERT **71.8**
  - XLM **71.0**
  - XLM-R (large) **76.1**
  - MMTE 61.9
  - English in-language reference: mBERT 85.2, XLM-R 84.7

  — [XTREME paper, Table 21](https://arxiv.org/pdf/2003.11080)
- **WikiANN / PAN-X:** Pan et al. 2017 is the source of PanX, which Altinok also uses for training spaCy NER. — [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf)

### Inferences
- The two "WikiANN Turkish" numbers measure different things. BERTurk's ~93.9 comes from in-language fine-tuning on the Turkish split, while XTREME's 71.8 (mBERT) and 76.1 (XLM-R) are zero-shot transfer from English. They should not be presented as a head-to-head comparison.
- On the Tür et al. news benchmark the progression is roughly 91.56 MUC (HMM, 2003), 91.94 CoNLL / 94.59 MUC (CRF, 2012), 93.59 F1 (BiLSTM-CRF with morphological embeddings, 2017), and 95.95 F (transformer+CRF, 2020). Metrics and splits vary between papers.

### Gaps
- In-language fine-tuned mBERT and XLM-R F1 on Turkish WikiANN was not found in a verified primary source within the search budget. The Rahimi et al. (2019) balanced WikiANN train/dev/test splits used by HF "wikiann/tr" were not verified here.
- The exact dataset behind Aras et al.'s 95.95% was not confirmed from the abstract. It is likely the Tür et al. news set, but that is unverified.
- No verified BERTurk score on the Tür et al. news data was retrieved.

## Q4. Linguistic challenges of Turkish for NER cited in the literature

### Takeaway
The literature consistently cites four challenges. Agglutination causes data sparsity, so lemma and morphological features are needed. Proper-noun suffixes are separated by an apostrophe, which helps on edited text but is often missing in tweets. Free word order means position gives no cue. Capitalization is a strong cue that breaks down in user-generated text. Vowel harmony is not discussed as an NER challenge in any source I retrieved.

### Cited Findings
- **Agglutination and sparsity:** Tür et al. (2003) say that "using the surface forms of the words results in data sparseness" and that Turkish can "produce thousands of forms (or even millions (Hankamer 1989)) for a given root." Vocabulary size dropped "on the order of 5" when using roots instead of surface forms. — [Tür et al. 2003](https://www.andrew.cmu.edu/user/ko/downloads/Papers/InformationExtraction.pdf)
- **Inflection example:** Şeker & Eryiğit describe Turkish as "a morphologically rich and highly agglutinative language," noting that *gitmek* "may appear in hundreds of different surface forms" against 5 in English. — [Şeker & Eryiğit 2012](https://aclanthology.org/C12-1150.pdf)
- **Apostrophe:** Şeker & Eryiğit write that "for the proper nouns, in formal texts, the inflectional suffixes are separated from the lemma by an apostrophe." They split such tokens in two, because the suffixes "are not part of the named entities." Stemming of the surrounding words still matters. — [Şeker & Eryiğit 2012](https://aclanthology.org/C12-1150.pdf)
- **Apostrophe in tweets:** Küçük et al. list "not separating names from suffixes with apostrophes" among the Turkish-specific errors in tweets. They also list missing capitalization (25.1% of PLOs), ASCII substitution of Turkish characters (ç, ğ, ı, ö, ş, ü), found in 10% of PLOs, and hashtag-merged names (7.6% of NEs). — [Küçük et al. 2014](https://aclanthology.org/L14-1328.pdf)
- **Free word order:** Şeker & Eryiğit write that "Turkish is a free word order language, so the position of the word in a sentence doesn't give us information about being a named entity or not." — [Şeker & Eryiğit 2012](https://aclanthology.org/C12-1150.pdf)
- **Ambiguous names and capitalization:** Şeker & Eryiğit note that Turkish first names are often common words (İpek "silk", Kaya "rock", Pembe "pink", Çiçek "flower"), and that "only the proper nouns and the initial words of the sentences start with an initial capital letter." — [Şeker & Eryiğit 2012](https://aclanthology.org/C12-1150.pdf)
- **Morphosyntax in general:** Altinok notes that "Turkish is an agglutinative language with complex morphology (Göksel and Kerslake, 2005), and its morphosyntactic characteristics are challenging to handle in NLP applications." — [Altinok 2023](https://aclanthology.org/2023.acl-long.768.pdf)
- **Morphology helps neural models too:** Güngör et al. show that adding morphological-analysis embeddings to a BiLSTM-CRF improves Turkish NER. — [arXiv 1706.00506](https://arxiv.org/abs/1706.00506)

### Inferences
- Vowel harmony matters for NER only indirectly: suffix allomorphs such as *-da/-de/-ta/-te* and *-ı/-i/-u/-ü* multiply surface variants of each inflected entity, which adds to the sparsity problem. That link is my own inference and was not found in a retrieved NER paper.
- Subword tokenization (WordPiece in BERTurk) is the modern answer to sparsity. Apostrophe-separated suffixes are generally split cleanly, while unapostrophized informal text remains hard.

### Gaps
- I found no retrieved Turkish NER paper that explicitly names vowel harmony as an NER challenge. To cite it, use a general Turkish grammar such as Göksel & Kerslake (2005), *Turkish: A Comprehensive Grammar*, which Altinok cites, and frame it as a morphological fact rather than an NER-literature claim.
- The Çöltekin et al. (2022) survey, "Resources for Turkish natural language processing: A critical survey" (Language Resources and Evaluation; [PMC9417072](https://pmc.ncbi.nlm.nih.gov/articles/PMC9417072/)), was blocked by a browser check, so its NER section was not read.
