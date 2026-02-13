# GhanaNouns

A lexicon of English noun phrases extracted from Ghanaian news, academic research, and speech data.  
The dataset provides a baseline vocabulary dataset for improving Machine Translation quality within the Ghanaian context.

---

## Rationale

Machine Translation systems often fail on region‑specific language varieties because they lack exposure to local vocabulary, collocations, and domains.  
Ghanaian English—while mutually intelligible with global English—exhibits distinct preferences in word usage, institutional references, and cultural concepts.

GhanaNouns, developed by Ghana NLP, addresses this gap by offering a high‑coverage, filtered set of noun phrases that appear naturally in Ghanaian news, academic writing, and speech.  
Our primary objectives are:

- Provide a baseline English‑noun lexicon sourced exclusively from authentic Ghanaian texts.
- Enable domain adaptation of MT models for Ghanaian English.
- Facilitate synthetic data generation (e.g., back‑translation, term‑augmented training) via frequency‑weighted vocabulary lists.
- Support human data collection (e.g., annotation, lexicon expansion) with a clean, deduplicated resource.
- Serve as a reference corpus for contrastive linguistic studies of Ghanaian vs. international English.

By releasing this dataset openly, we aim to lower the barrier for developing NLP tools that work for and with Ghanaian users.

---

## 🙋 Contributors

This project was a collaborative effort. We would like to thank the following volunteers who dedicated their time to creating the dataset:

1. [Jonathan Ato Markin](https://www.linkedin.com/in/atomarkin/)
2. [Emmanuel Saah](https://www.linkedin.com/in/emmanuel-saah/)
3. [Gerhardt Datsomor](https://www.linkedin.com/in/gerhardt-datsomor/)
4. [Kasuadana Sulemana Adams](https://www.linkedin.com/in/kasuadana1/)
5. [Lucas Kpatah](https://www.linkedin.com/in/lucas-kpatah-351086376/)
6. [Mich-Seth Owusu](https://www.linkedin.com/in/mich-seth-owusu/)

---

## 📊 Dataset Overview

| Metric                             | Value      |
|------------------------------------|------------|
| Total unique noun phrases          | **806,317**|
| … from news, research & speech     | 43,345     |
| … from news & research only        | 66,024     |
| … from news & speech only          | 28,226     |
| … from research & speech only      | 10,990     |
| … exclusively in news              | 395,534    |
| … exclusively in research          | 650,886    |
| … exclusively in speech            | 221,714    |
| Language‑filtered                  | FastText (lid.176, ≥0.7) |
| Minimum phrase length              | 1 word     |
| Maximum phrase length              | 6+ words   |

All phrases are **lowercased** and stripped of leading stopwords.  
Proper nouns, acronyms, and non‑alphabetic tokens are **removed** during extraction.

---

## 🔍 Sample Data

| phrase     | news_count | research_count | speech_count | news_%  | research_% | speech_% | avg_%  | source               |
|------------|------------|----------------|--------------|---------|------------|----------|--------|----------------------|
| people     | 109,037    | 50,895         | 145,181      | 0.9375  | 0.4683     | 2.3585   | 1.2548 | news_research_speech |
| government | 110,414    | 13,981         | 66,410       | 0.9493  | 0.1286     | 1.0788   | 0.7189 | news_research_speech |
| study      | 4,175      | 227,243        | 559          | 0.0359  | 2.0910     | 0.0091   | 0.7120 | news_research_speech |
| things     | 22,866     | 6,708          | 61,456       | 0.1966  | 0.0617     | 0.9984   | 0.4189 | news_research_speech |
| lot        | 19,615     | 8,939          | 45,384       | 0.1686  | 0.0823     | 0.7373   | 0.3294 | news_research_speech |
| money      | 29,007     | 13,381         | 34,107       | 0.2494  | 0.1231     | 0.5541   | 0.3089 | news_research_speech |
| president  | 19,220     | 370            | 46,600       | 0.1652  | 0.0034     | 0.7570   | 0.3085 | news_research_speech |
| place      | 32,504     | 17,320         | 25,863       | 0.2795  | 0.1594     | 0.4201   | 0.2863 | news_research_speech |
| work       | 25,582     | 34,592         | 17,722       | 0.2199  | 0.3183     | 0.2879   | 0.2754 | news_research_speech |
| terms      | 10,825     | 21,156         | 26,812       | 0.0931  | 0.1947     | 0.4356   | 0.2411 | news_research_speech |
| person     | 17,642     | 9,679          | 28,029       | 0.1517  | 0.0891     | 0.4553   | 0.2320 | news_research_speech |
| law        | 25,990     | 3,630          | 25,627       | 0.2235  | 0.0334     | 0.4163   | 0.2244 | news_research_speech |
| party      | 30,833     | 1,762          | 22,335       | 0.2651  | 0.0162     | 0.3628   | 0.2147 | news_research_speech |
| …          | …          | …              | …            | …       | …          | …        | …      | …                    |

*Percentages are normalised within each source corpus.*

---

## 🧱 File Format

**`ghana-nouns.csv`**  
UTF‑8, comma‑separated, header row.

| Column                | Description |
|-----------------------|-------------|
| `phrase`              | Lowercased noun phrase |
| `news_count`          | Raw frequency in the news corpus |
| `research_count`      | Raw frequency in the research corpus |
| `speech_count`        | Raw frequency in the speech corpus |
| `news_percentage`     | Relative frequency within news noun‑phrase tokens (×100) |
| `research_percentage` | Relative frequency within research noun‑phrase tokens (×100) |
| `speech_percentage`   | Relative frequency within speech noun‑phrase tokens (×100) |
| `average_percentage`  | Arithmetic mean of the available source percentages |
| `source`              | Combination of one or more of: `news`, `research`, `speech` |

---

## ⚙️ Methodology (Summary)

1. **Sentence collection**  
   - 2.3M sentences from Ghanaian online news (2018–2024).  
   - 2.7M sentences from Ghana‑focused academic publications.  
   - Additional sentences from Ghanaian speech data.

2. **Noun phrase extraction** (`extract_np.py`)  
   - spaCy `en_core_web_sm`, GPU accelerated.  
   - Keep only **all‑lowercase** phrases.  
   - Strip leading stopwords.  
   - Deduplicate and count.

3. **Cleaning & merging** (`combine-all.py`)  
   - Remove non‑alphabetic characters.  
   - Remove all‑caps / multi‑capitalised tokens.  
   - Filter out adjectives (POS tagging).  
   - Merge news, research & speech counts.

4. **Language identification** (`filter-non-english.py`)  
   - FastText `lid.176.bin`, confidence ≥ 0.7.  
   - Retained **56.9%** of phrases as English.

---

## 🚀 Usage Ideas

### • Machine Translation adaptation  
Use the frequency distributions to **bias subword tokenisation** or to create **domain‑adapted vocabularies** for finetuning MT models (e.g., M2M100, NLLB, OPUS‑MT).

### • Synthetic data generation  
- **Term injection**: Replace general English nouns in parallel sentences with Ghanaian‑specific terms from the dataset.  
- **Back‑translation**: Use the phrase list as a target‑side lexicon to guide back‑translation from English into Ghanaian languages.  
- **Masked language modelling**: Pretrain a language model on Ghanaian English texts, then evaluate its lexical knowledge using this dataset.

### • Human data collection  
- **Annotation tasks**: Use the cleaned phrases as a starting pool for collecting translations into Ghanaian languages (Twi, Ga, Ewe, etc.) or for sentiment / topic labelling.  
- **Lexical resource expansion**: Crowdsource synonyms or regional variants based on the core list.

### • Linguistic analysis  
- Compare relative frequencies of common nouns across news, academic, and speech registers.  
- Identify terms that are **overrepresented** in Ghanaian English compared to general corpora (e.g., COCA, BNC).

---

## 📦 Repository Contents

```
.
├── data/
│   └── ghana-nouns.csv   # Main dataset
├── scripts/
│   ├── extract_np.py          # Noun phrase extraction
│   ├── combine-all.py         # Merge, clean, filter adjectives
│   ├── filter-non-english.py  # FastText language filtering
├── README.md
└── LICENSE
```

---

## 🏛️ About Ghana NLP

Ghana NLP is an open‑source community initiative focused on building natural language processing resources and tools for the languages of Ghana.  
We develop datasets, models, and software to promote research and applications in Ghanaian languages and Ghanaian English.  
Our work is entirely volunteer‑driven and publicly released under open licenses.

- 🌐 [ghananlp.org](https://ghananlp.org)  
- 🐦 [@GhanaNLP](https://twitter.com/GhanaNLP)  
- 💻 [GitHub](https://github.com/ghananlp)

---

## 📖 Citation

If you use GhanaNouns in your research or applications, please cite:

```
Ghana NLP. (2025). GhanaNouns: A corpus of noun phrases from Ghanaian news and academic texts.
[Data set]. https://github.com/ghananlp/GhanaNouns
```

BibTeX:
```bibtex
@misc{ghananlp2025ghananouns,
  title = {GhanaNouns: A corpus of noun phrases from Ghanaian news and academic texts},
  author = {{Ghana NLP}},
  year = {2025},
  howpublished = {\url{https://github.com/ghananlp/GhanaNouns}},
}
```

---

## 📄 License

**Creative Commons Attribution 4.0 International (CC BY 4.0)**  
You are free to share and adapt the material for any purpose, even commercially, provided appropriate credit is given.

---

## 🙋 Contact

We welcome contributions, bug reports, and suggestions via [GitHub Issues](https://github.com/ghananlp/GhanaNouns/issues).  
For general inquiries: **info@ghananlp.org**  

If you extend the dataset or apply it in an interesting way, please let us know—we'd love to feature your work!

---

*Built with spaCy, FastText, and a lot of Ghanaian text.*  
**🇬🇭 Made with ❤️ by Ghana NLP.**
