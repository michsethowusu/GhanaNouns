# GhanaNouns: A Large-Scale Dataset of Ghanaian Nouns

This repository contains a curated dataset of concrete nouns extracted from over 2 million words and phrases found in Ghanaian news articles. This resource is provided by Ghana NLP to improve the quality of Machine Translation for vocabulary used in the Ghanaian context.

## Overview

In low-resource language contexts, nouns are essential for clear communication but standard Machine Translation models are yet to catchup with accurate translation for these diverse and yet important part of speech. 

By providing this dataset, our goal is to establish a baseline that can be used to collect accurate translations of vocabulary used in the Ghanaian context.

 

## Methodology

The dataset was produced through a multi-stage pipeline:

1. **Initial Extraction:** 2 million potential nouns were identified from Ghanaian news archives using spaCy.
2. **LLM Refinement:** Because rule-based POS tagging can be imprecise, we used the Mistral AI API to verify and filter the list.
3. **Cleaning:** The final set was filtered to exclude abstract concepts, keeping only common concrete nouns.
4. **Frequency Analysis:** Nouns were mapped against their original occurrence counts to determine their relevance.

## Dataset Structure

The output is provided in multiple CSV files, categorized by their minimum frequency of appearance. This allows users to choose between a broad vocabulary or a highly-vetted "core" terminology list.

| File                | Threshold         | Usage Case                                          |
| ------------------- | ----------------- | --------------------------------------------------- |
| `nouns_min_2.csv`   | ≥ 1 occurrence    | Comprehensive vocabulary for research.              |
| `nouns_min_10.csv`  | ≥ 10 occurrences  | Standard common nouns for general NLP tasks.        |
| `nouns_min_50.csv`  | ≥ 50 occurrences  | High-confidence terminology for MT anchoring.       |
| `nouns_min_500.csv` | ≥ 500 occurrences | Core concrete nouns essential for every dictionary. |

### CSV Format

Each file contains the following columns:

- `noun`: The cleaned concrete noun (English).
- `frequency`: Total number of occurrences in the original 2-million-word corpus.

## Contributors

This project was a collaborative effort. We would like to thank the following volunteers who dedicated their time in creating the dataset:

1. [Jonathan Ato Markin](https://www.linkedin.com/in/atomarkin/)
2. [Emmanuel Saah](https://www.linkedin.com/in/emmanuel-saah/)
3. [Gerhardt Datsomor](https://www.linkedin.com/in/gerhardt-datsomor/)
4. [Kasuadana Sulemana Adams](https://www.linkedin.com/in/kasuadana1/)
5. [Lucas Kpatah](https://www.linkedin.com/in/lucas-kpatah-351086376/)
6. [Mich-Seth Owusu](https://www.linkedin.com/in/mich-seth-owusu/)

## Acknowledgments

This project was made possible by the volunteers of Ghana NLP who contributed their time and compute resources to process the 2-million-word dataset using the Mistral API. We also thank the developers of Mistral for building a great model and making their API accessible to developers in Africa. 

## About Ghana NLP

Ghana NLP is an Open Source initiative dedicated to the development of NLP tools for Ghanaian languages and the advancement of African Language Technology.
