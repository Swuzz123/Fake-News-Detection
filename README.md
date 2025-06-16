# Fake News Detection
## Table of Contents
- [Introduction](#introduction)
  - [Data Sources](#data-sources)
  - [Architecture](#architecture)
  - [Prerequisites](#prerequisites)
- [Overview](#overview)
  - [Data Preprocessing](#data-preprocessing)
    - [1. Remove unnecessary columns and Label for the data](#remove-unnecessary-columns-and-label-for-the-data)
    - [2. Check missing and duplicated values](#check-missing-and-duplicated-values)
    - [3. Text cleaning and Normalize text to correct format](#text-cleaning-and-normalize-text-to-correct-format)
  - [Explore Data Analysis (EDA)](#explore-data-analysis-eda)
  - [Model Building and Training](#model-building-and-training)
    - [1. Build with Machine Learning Algorithms](build-with-machine-learning-algorithms)
    - [2. Build with Word2Vec and LSTM](#build-with-word2vec-lstm)
    - [3. Fine tuning with BERT](#fine-tuning-with-bert)
  - [Performance Evaluation](#performance-evaluation)
  - [Model Deployment](#model-deployment)
 
## Introduction
The widespread dissemination of fake news and propaganda presents serious societal risks, including the erosion of public trust, political polarization, manipulation of elections, and the spread of harmful misinformation during crises such as pandemics or conflicts.
From an NLP perspective, detecting fake news is fraught with challenges. Linguistically, fake news often mimics the tone and structure of legitimate journalism, making it difficult to distinguish using surface-level features. 
The absence of reliable and up-to-date labeled datasets, especially across multiple languages and regions, hampers the effectiveness of supervised learning models. 
Additionally, the dynamic and adversarial nature of misinformation means that malicious actors constantly evolve their language and strategies to bypass detection systems. Cultural context, sarcasm, satire, and implicit bias further complicate automated analysis. 
Moreover, NLP models risk amplifying biases present in training data, leading to unfair classifications and potential censorship of legitimate content. 
These challenges underscore the need for cautious, context-aware approaches, as the failure to address them can inadvertently contribute to misinformation, rather than mitigate it.

### Data Sources
- **True Articles:** Reputable media outlets like Reuters, The New York Times, The Washington Post, etc.
- **Fake/Misinformation/Propaganda Articles:**
  - American right-wing extremist websites (e.g., Redflag Newsdesk, Breitbart, Truth Broadcast Network)
  - Public dataset from: Ahmed, H., Traore, I., & Saad, S. (2017): "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques" *(Springer LNCS 10618)*
 
### Architecture

![My image](./images/Architecture_Fake_News_Detection.png)

### Prerequisites
1. Python 3.10.8
  - This setup requires that your machine has python 3.10.8 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in how to run software section). To do that check this:
  https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.
2. After installing python, you also need to download the required packages
```text
pip install -r requirments.txt
```

## Overview
### Data Preprocessing
#### 1. Removing unneccessary columns and Labeling for the data
- There was a column `Unnamed: 0` that did not serve for any further purpose, so I decided to delete it.
- The original dataset had not been labeled as **Fake** or **Real**, so I labeled for the dataset with **Fake News** being 0 and **Real News** being 1

#### 2. Checking missing and duplicated values
After checking missing values, I noticed that having **29 missing values**. Furthermore, I analysized clearly texts, which maybe have some cases relating **blank space**, can not be understood by computer so I wrote an extra function to check the texts in **blank space** form and they are counted as missing values.

```python
# Function to convert blank space to null value
def blank_to_nan(df):
    tmp = []
    for item in df:
        if item == '':
            tmp.append(np.nan)
        else:
            tmp.append(item)
    return tmp
```

Next, I checked duplicated values and the recorded result showed there were **9988 duplicated values**

#### 3. Text Cleaning and Normalize text to correct format
Computers can not learn all the words from large text passages if they contain too many irrelevant words or characters. Therefore, it is necessary to preprocess the text by removing non-text characters and filtering out potentially noisy words—those that appear frequently but carry little analytical value, such as “the,” “and,” or “of.”

In addition, to help the model better understand the actual content of the text, I need to reduce words to their base form—this is where lemmatization becomes essential. Instead of having the model learn separate variations of a word like “running,” “ran,” or “runs,” I standardized them into a single meaningful root form. This reduced the number of words the model needs to learn and improved its generalization ability.

```python
# Import processing text libraries
import re
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)  # Remove URLs/HTML
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # Keep only letters
    text = unidecode(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)
```

### Explore Data Analysis (EDA)
#### 1. View Label Distribution
Here, I conducted an analysis to evaluate the distribution of Fake and True News in order to determine whether the dataset was balanced. The results showed that the dataset was perfectly balanced, with True News accounting for 50.33% and Fake News accounting for 49.67%.

![label_distribution](https://github.com/Swuzz123/Fake-News-Detection/blob/master/images/Frequency%20of%20Fake%20vs%20True%20News.png)

#### 2. Text Length Analysis
In this step, I measured the text length (in terms of words, characters, or sentences) to understand the data structure, identify differences between the classes (Fake vs. True), optimize the `max_length` parameter for tokenization, detect anomalies, and support the creation of additional features for the model. After analyzing text length, I found that **Fake news** was generally shorter, though a few exceptions were unusually long; **True news** tended to be longer and more consistent in length. Text length proved to be a potentially useful feature for distinguishing between the two classes.

![text_length](https://github.com/Swuzz123/Fake-News-Detection/blob/master/images/Number%20of%20words%20in%20the%20text%20(Fake%20vs%20True).png)

#### 3. Top Words that Appear Frequently
I analyzed the most frequently occurring words (Top Words) in the text data to identify key terms that reflected the main topics or linguistic trends. The goal was to understand the prominent content and compare the differences between the classes (Fake vs. True). It was observed that **Fake news** often focused on specific individuals (Trump, Clinton, Obama) and hot-button events (war, election, Russia), which may suggest that **Fake news** tended to use dramatic language or highlight personal/sensational stories to capture attention. In contrast, **True news** showed a tendency to use more general descriptive language (state, people, government) and terms related to data or official information (percent, law, leader, official), indicating that **True news** was more often grounded in reports, statistics, or official information from institutions or government sources.

![worldcloud_fake_news](https://github.com/Swuzz123/Fake-News-Detection/blob/master/images/WordCloud%20of%20top%20words%20appear%20most%20-%20Fake%20News.png)

![worldcloud_true_news](https://github.com/Swuzz123/Fake-News-Detection/blob/master/images/WordCloud%20of%20top%20words%20appear%20most%20-%20True%20News.png)


#### 4. Contextual Analysis
I examined whether words like “war” and “election” in **Fake news** appeared in accusatory sentences, and whether words like “law” and “percent” in **True news** were related to official reports. The results showed that the number of sentences reflecting official reporting in **True news** was twice as high as in **Fake news**, suggesting that accurate information often relied on authoritative sources. In contrast, the number of such sentences in **Fake news** was significantly lower, implying that misleading content tended to use more critical or accusatory language.

![contextual_analysis](https://github.com/Swuzz123/Fake-News-Detection/blob/master/images/The%20number%20of%20accusatory%20(False)%20statements%20VS%20the%20official%20report%20(True).png)

### Model Building and Training
#### 1. Build with Machine Learning Algorithms
**Step 1: Selecting the best N-gram range**<br>
In this step, I selected the optimal N-gram range (from unigrams to trigrams) to prepare the data for vectorization using the TF-IDF Vectorizer in the following stages. The goal was to choose the N-gram range that yielded the highest F1-score, ensuring that the extracted features were suitable for text representation in the Fake News Classification task. The results showed that the best-performing N-gram range was bigrams (1, 2).

![best_n_gram_range](./images/Best_N_gram_range.png)

**Step 2: Vectorizing Text with TF-IDF and Integrating Semantic Features into the Training Set** <br>
In this step, I used **TF-IDF (Term Frequency–Inverse Document Frequency)** as the main vectorization technique to convert text into a structured numerical format suitable for machine learning algorithms. In addition to TF-IDF, I incorporated other semantic-based features to capture different linguistic and contextual characteristics of the text. Specifically, I added the following three additional semantic features:
- **Average Sentiment Polarity:** This score reflects the overall emotional tone of the text, classifying it as positive, neutral, or negative. Sentiment analysis is especially important in detecting fake news, which often exploits strong emotional triggers to mislead readers. The polarity score provides valuable insights into the emotional intent of the content.

- **Readability Score (Flesch–Kincaid):** This metric evaluates the complexity of a text. Texts that are overly simplistic or excessively complex may signal fake news. The score estimates how easy a passage is to read based on sentence length and word complexity, helping to identify content designed to mislead or manipulate through overly technical or overly simplistic language.

- **Thematic Diversity:** This feature analyzes the variety of topics covered within a text to better understand the spread and focus of its content. Low thematic diversity may indicate fake news, which often recycles a narrow set of biased or emotionally charged topics.

![formula_3_semantic_features](./images/Formula_3_semantic_features.png)

After applying TF-IDF vectorization, the resulting sparse matrix is already normalized and its values lie within the range [0,1]. However, the three additional semantic features are not on the same scale:

- **Sentiment Polarity:** Values can be both negative and positive, and typically fall within a small numerical range.

- **Readability Score:** The values can be significantly higher than those from the TF-IDF matrix.

- **LDA Topic Vector:** These are probabilities, so they naturally lie between 0 and 1, but may still need scaling to match the magnitude of TF-IDF.

To ensure consistency and effective integration with the TF-IDF features, these semantic features should be rescaled or normalized before being combined into the final feature set.

**Step 3: Make a Pipeline for training different models** <br>





