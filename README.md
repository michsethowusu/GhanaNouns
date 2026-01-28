# GhanaNouns: Multilingual Dataset of Ghanaian Nouns

This is an onoing project to create a dataset of nouns that are relevant to the Ghanaian context and translate them to Ghanaian languages using Large Language Models that have strong performance for Ghanaian languages. There are currently 875,476 nouns that have been extracted from various sources and translated to languages [specified below](#completed-languages).

## Dataset Structure

The dataset is stored as a CSV file. Each row represents an English noun or noun phrase translated under a specific domain. Translations into Ghanaian languages are added incrementally as they become available.

### Columns

- **`text`** – English noun or noun phrase
- **`domain`** – Translation context (e.g. `general`, `agric`)
- **`translation_<lang>`** – Translation in a Ghanaian language, where `<lang>` is the 3-letter ISO language code. These columns are added as new translations become available.

### Example

| text   | domain  | translation_twi | translation_ewe |
| ------ | ------- | --------------- | --------------- |
| bank   | general | sikakorabea     | gaƒoƒo          |
| bank   | agric   | nsu ho asase    | tɔdziƒe         |
| maize  | agric   | aburo           | bli             |
| market | general | adwumam         | asi             |

## Dataset Status

The dataset is currently being translated across a wide range of Ghanaian languages using Google Gemini API which has proven to be the strongest model for Ghanaian languages. Accuracy varies by language, and we don't expect the translation to be perfect. For detailed metrics on estimated overall accuracy, please refer to the results from the [Nsanku Project](https://github.com/GhanaNLP/nsanku?tab=readme-ov-file#language-specific-results). 

### Completed Languages

| Language | Status      |
| -------- | ----------- |
| Twi      | ✅ Completed |

### Languages In Progress

The following languages are currently in the queue for translation. We welcome contributors to help move these to the completed list.

| Language         | Status        | Language         | Status        |
| ---------------- | ------------- | ---------------- | ------------- |
| Abron            | ⏳ Coming Soon | Mampruli         | ⏳ Coming Soon |
| Gikyode          | ⏳ Coming Soon | Deg              | ⏳ Coming Soon |
| Dangme           | ⏳ Coming Soon | Nawuri           | ⏳ Coming Soon |
| Siwu             | ⏳ Coming Soon | Chumburung       | ⏳ Coming Soon |
| Anyin            | ⏳ Coming Soon | Nkonya           | ⏳ Coming Soon |
| Avatime          | ⏳ Coming Soon | Delo             | ⏳ Coming Soon |
| Bisa             | ⏳ Coming Soon | Nyagbo           | ⏳ Coming Soon |
| Bimoba           | ⏳ Coming Soon | Nzema            | ⏳ Coming Soon |
| Southern Birifor | ⏳ Coming Soon | Esahie           | ⏳ Coming Soon |
| Tuwuli           | ⏳ Coming Soon | Paasaal          | ⏳ Coming Soon |
| Ntcham           | ⏳ Coming Soon | Tumulung Sisaala | ⏳ Coming Soon |
| Buli             | ⏳ Coming Soon | Selee            | ⏳ Coming Soon |
| Anufo            | ⏳ Coming Soon | Tafi             | ⏳ Coming Soon |
| Dagbani          | ⏳ Coming Soon | Tampulma         | ⏳ Coming Soon |
| Southern Dagaare | ⏳ Coming Soon | Vagla            | ⏳ Coming Soon |
| Ewe              | ⏳ Coming Soon | Konkomba         | ⏳ Coming Soon |
| Fante            | ⏳ Coming Soon | Kasem            | ⏳ Coming Soon |
| Ga               | ⏳ Coming Soon | Farefare         | ⏳ Coming Soon |
| Gonja            | ⏳ Coming Soon | Hanga            | ⏳ Coming Soon |
| Konni            | ⏳ Coming Soon | Kusaal           | ⏳ Coming Soon |
| Lelemi           | ⏳ Coming Soon | Sekpele          | ⏳ Coming Soon |

## How to Contribute

We are looking for volunteers to help complete the translations for the languages listed above. If you would like to contribute, please follow these steps:

1. **Identify an Incomplete Language:** Check the "Languages In Progress" section above or look into the language folders to see which nouns have not yet been translated.
2. **Locate the Scripts:** Navigate to the `translation_scripts/` folder in this repository.
3. **Run the Translation Script:**
   - Choose the incomplete language you wish to work on.
   - Execute the translation script provided in the folder for the language you want to create a translation for (ensure you have the Google Gemini API Access).
5. **Submit Your Contribution:**
   - Add the completed translation file to the `contributions/` directory.
   - Open a **Pull Request** (PR) to submit the dataset.


## Contributors

This project is a collaborative effort. We would like to thank the following volunteers who have dedicated their time to creating this resource:

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
