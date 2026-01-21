# GhanaNouns: A Large-Scale Dataset of Ghanaian Nouns

This repository contains a curated dataset of nouns extracted from over 2 million words and phrases found in Ghanaian news articles. This resource is provided by Ghana NLP to improve the quality of Machine Translation for vocabulary used in the Ghanaian context.

## Overview

In low-resource language contexts, nouns are essential for clear communication, but standard Machine Translation models have yet to catch up with accurate translations for this diverse and important part of speech.

By providing this dataset, our goal is to establish a baseline that can be used to collect accurate translations of vocabulary specifically relevant to the Ghanaian context.

## Methodology

The dataset was produced through a multi-stage pipeline:

1. **Initial Extraction:** 2 million potential nouns were identified from Ghanaian news archives using spaCy.
2. **LLM Refinement:** Because rule-based POS tagging can be imprecise, we used the Mistral AI API to verify and filter the list.
3. **Cleaning:** The final set was filtered to exclude abstract concepts, keeping only common concrete nouns.
4. **Frequency Analysis:** Nouns were mapped against their original occurrence counts to determine their relevance.
5. **Machine Translation:** Translations into various Ghanaian languages were generated using Google Gemini 3 Flash.

## Dataset Structure

The dataset has been reorganized to provide a single source of truth for frequencies and individual translation files for a wide array of Ghanaian languages.

### 1. Master Frequency List

- **File:** `nouns_master_list.csv`
- **Description:** A comprehensive file containing all verified nouns along with their frequency counts from the 2-million-word corpus. This allows researchers to filter the data based on their own custom thresholds.
- **Columns:**
  - `noun`: The cleaned concrete noun (English).
  - `frequency`: Total number of occurrences.

### 2. Language-Specific Lists

We provide individual noun lists for the following Ghanaian languages to facilitate human evaluation and other NLP use cases.

**Note on Translation Quality:** The quality of these translations may differ significantly based on the specific language. For detailed information on the estimated overall accuracy for each language, please refer to the results from the [Nsanku Project](https://github.com/GhanaNLP/nsanku?tab=readme-ov-file#language-specific-results).

|                  |          |                  |                  |
| ---------------- | -------- | ---------------- | ---------------- |
| Abron            | Gikyode  | Dangme           | Siwu             |
| Anyin            | Avatime  | Bisa             | Bimoba           |
| Southern Birifor | Tuwuli   | Ntcham           | Buli             |
| Anufo            | Dagbani  | Southern Dagaare | Ewe              |
| Fante            | Ga       | Gonja            | Farefare         |
| Hanga            | Konni    | Kusaal           | Lelemi           |
| Sekpele          | Mampruli | Deg              | Nawuri           |
| Chumburung       | Nkonya   | Delo             | Nyagbo           |
| Nzema            | Esahie   | Paasaal          | Tumulung Sisaala |
| Selee            | Tafi     | Tampulma         | Twi              |
| Vagla            | Konkomba | Kasem            |                  |

## Contributors

This project was a collaborative effort. We would like to thank the following volunteers who dedicated their time to creating the dataset:

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
