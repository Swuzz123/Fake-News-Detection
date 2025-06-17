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
  - [Model Deployment](#model-deployment)
  - [Conclusion](#conclusion)
 
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

**Step 3: Build a Unified Training Pipeline for Multiple Models** <br>

In this step, I created a unified pipeline that allows easy switching between different machine learning models. With this setup, I can simply call a function and specify the model name I want to test, and the pipeline will handle all the necessary preprocessing, training, and evaluation steps automatically.

```python
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score

# Make a list to store all the names of model training and accuracy
models = []
accuracy = []
f1_train = []
f1_test = []

# Build model function
def build_model(ml_model, X_train, y_train, X_test, y_test, accuracy, f1_train, f1_test):
    # Fit the model
    ml_model.fit(X_train, y_train)
    
    # Calculate scores 
    y_pred = ml_model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    f1_test.append(f1_score(y_test, y_pred))
    f1_train.append(f1_score(y_train, ml_model.predict(X_train)))
    
    # Print classification report
    print(classification_report(y_test, y_pred))
```

This is the result after training and Support Vector Machine (SVM) is the model with best performance as well as the most stable among 4 remaining models

![training_result](./images/Training_result.png)

#### 2. Build with Word2Vec and LSTM
**Step 1: Generating Word Embeddings with Word2Vec**<br>

In this step, I fed the preprocessed text data into the Word2Vec model to learn vector representations (embeddings) for each word. The Word2Vec model captures semantic relationships between words by placing those with similar contexts closer together in the vector space. This embedding can later be used to enhance the input features for downstream classification tasks.

```python 
import gensim 

EMBEDDING_DIM = 100
w2v_model = gensim.models.Word2Vec(sentences = X, vector_size = EMBEDDING_DIM, window = 10, min_count = 1)
```

**Step 2: Text Tokenization and Sequence Padding**<br>

In this step, each sentence was converted into a sequence of integer indices corresponding to words in the vocabulary built from the training data. Then, padding was applied to standardize the length of all sequences to a fixed size `(maxlen)`. If a sentence was shorter than `maxlen`, zeros were appended to the end (post-padding). If it was longer, the extra tokens at the end were truncated. This normalization step was necessary to ensure that all input data had a uniform shape, which is required for feeding into Deep Learning models.

```python 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization
tokenizer = Tokenizer(num_words = 8000, oov_token = '<OOV>')
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)

# Padding
maxlen = 700

X = pad_sequences(X, maxlen = maxlen)
```

**Step 3: Constructing Embedding Matrix for Tokenized Vocabulary (Embedding Matrix)**<br>

In this step, I used a pre-trained Word2Vec model to build the embedding matrix, where each row corresponds to a word in the tokenizer's vocabulary and contains its associated semantic embedding vector. This matrix was later used to initialize the embedding layer in deep learning model, allowing them to start with rich, pre-learned word representations instead of learning from scratch.

```python
# Function to create weight maxtrix from Word2Vec model
def get_weight_matrix(model, vocab):
    # Total vocab size + 0 for unknown words
    vocab_size = len(vocab) + 1
    
    # Define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    
    # Step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        if word in model.wv:
            weight_matrix[i] = model.wv[word]
        
    return weight_matrix

embedding_vectors = get_weight_matrix(w2v_model, word_index)
```

**Step 4: Building and Compiling the LSTM Model with Pre-trained Embedding Layer**<br>

In this step, I built a sequential LSTM model for binary text classification. The model began with a non-trainable embedding layer, which used the pre-trained Word2Vec embedding matrix to represent each word as a dense vector. I then added an LSTM layer with 128 units to capture sequential dependencies in the text, followed by a dense output layer with a sigmoid activation function for binary classification. The model was compiled using the Adam optimizer and binary cross-entropy loss function, and was evaluated using accuracy as the performance metric.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()

# Non-trainable embedding layer
model.add(Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, weights = [embedding_vectors], input_length = maxlen, trainable = False))

# LSTM 
model.add(LSTM(128, return_sequences = False))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.build(input_shape=(None, maxlen))

model.summary()
```

After completing the training step, here are the results I obtained.

![accuracy_LSTM](./images/Accuracy_Word2Vec_LSTM.png)

#### 3. Fine tuning with BERT

In this step, I performed fine-tuning using a pre-trained BERT model. Since BERT is a large and computationally intensive model, training it on a CPU would be extremely slow. Therefore, I recommend running the training process on Kaggle or Google Colab, which both offer free GPU support, or on your own machine if it has a GPU available. If you're interested in the full implementation details, you can download the notebook from the notebook https://github.com/Swuzz123/Fake-News-Detection/blob/master/notebook/fine_tuning_with_BERT_model.ipynb provided and try it yourself. I have included clear explanations for every step, so you can better understand how BERT was fine-tuned for the Fake News Classification task.

### Model Deployment

*Update soon!*

### Conclusion

#### 1. Traditional Machine Learning Models (e.g., SVM, Logistic Regression):

After preprocessing and feature extraction (e.g., TF-IDF), these models performed well on balanced and cleaned datasets. Among them, Support Vector Machine (SVM) achieved the best and most stable performance with **94% accuracy**.

- Strengths: Fast training and inference, simple implementation, good performance on moderately sized datasets.

- Weaknesses: Unable to capture word semantics; limited adaptability to evolving fake news patterns; performance may drop with noisy or biased data.

#### 2. Word2Vec + LSTM Model:

Combining Word2Vec embeddings with LSTM allowed the model to capture long-term dependencies and subtle patterns. It achieved **96% accuracy** on the training set.

- Strengths: Captures contextual patterns and semantic meaning; effective on longer, more complex texts.

- Weaknesses: Resource-intensive; requires careful hyperparameter tuning; risk of overfitting without sufficient diverse data.

#### 3. Fine-Tuned BERT Model:

Leveraging the power of Transformers and contextualized embeddings, the fine-tuned BERT model significantly outperformed the other methods, achieving up to **99% accuracy**.

- Strengths: Superior understanding of language context and semantics; handles nuanced expressions and long texts well; pre-trained knowledge makes it robust.

- Weaknesses: High computational cost; requires GPU; complex tuning; risk of overfitting on small or biased datasets.

## Reference

https://memart.vn/tin-tuc/blog/tat-tan-tat-ve-lda-la-gi-va-ung-dung-trong-bai-toan-machine-learning-vi-cb.html
https://www.mdpi.com/2078-2489/16/3/189
https://www.sfu.ca/~mtaboada/docs/publications/Asr_Mokhtari_Taboada.pdf
https://npl0204.github.io/projects/Fake-News-Detection/
https://towardsdatascience.com/leveraging-n-grams-to-extract-context-from-text-bdc576b47049/
https://www.sciencedirect.com/org/science/article/pii/S1546221823006380
https://www.researchgate.net/publication/360057865_Fake_news_detection_using_deep_learning
https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4