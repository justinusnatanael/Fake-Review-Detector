# Fake-Review-Detector
# Fake Review Detection for Tokopedia Product Reviews Using Deep Learning
### Bachelor Thesis Project

## Overview
This project was developed as part of my bachelor's thesis in the Data Science Program at Bina Nusantara University. The research focuses on detecting fake product reviews in Indonesian e-commerce platforms using deep learning techniques and text classification methods.

The study uses review data collected from Tokopedia and evaluates multiple deep learning architectures to identify the most effective and computationally efficient model for fake review detection in real-time environments.

In addition to model development, this thesis introduces a hybrid labeling pipeline that combines human annotation, rule-based criteria, and Large Language Model (LLM)-assisted labeling to create a culturally relevant Indonesian fake review dataset.

---

## Thesis Objectives
The primary objectives of this research are:

- Develop an automated fake review detection system for Indonesian e-commerce reviews.
- Compare the performance of several deep learning architectures for text classification.
- Analyze the trade-off between model accuracy and inference speed.
- Build a lightweight and scalable solution suitable for real-time deployment.
- Explore hybrid human-LLM labeling strategies for dataset creation.

---

## Dataset
The dataset was collected from Tokopedia using web scraping techniques with Selenium and BeautifulSoup.

### Collected Features
- Review text
- Star rating
- Helpful vote count

### Product Categories
- Electronics
- Beauty products

### Dataset Statistics
- Total reviews: 7,875
- Real reviews: ~82.9%
- Fake reviews: ~17.1%

---

## Technologies & Tools
- Python
- TensorFlow / Keras
- Scikit-learn
- Selenium
- BeautifulSoup
- Pandas
- NumPy
- Matplotlib

---

## Research Methodology

### 1. Data Collection
- Web scraping from Tokopedia review pages
- Consumer perception survey using Google Forms
- Anti-bot scraping mechanisms:
  - Rotating user-agents
  - Random request delays
  - Headless browser rendering

### 2. Data Preprocessing
- Text cleaning and normalization
- Stopword removal
- Tokenization
- Indonesian stemming using Sastrawi
- TF-IDF vectorization
- Feature scaling for numerical features

### 3. Hybrid Labeling Pipeline
The labeling process combines:
- Human annotation
- Rule-based fake review criteria
- LLM-assisted automated labeling using prompt engineering

### 4. Deep Learning Models
Several architectures were implemented and compared:
- Deep Neural Network (DNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (Bi-LSTM)
- Gated Recurrent Unit (GRU)
- Convolutional Neural Network (CNN)

### 5. Evaluation Metrics
The models were evaluated using:
- PR-AUC
- Macro F1-Score
- Matthews Correlation Coefficient (MCC)
- Accuracy
- Inference speed

---

## Results

| Model | PR-AUC | Macro F1 | MCC | Speed (ms) |
|---|---|---|---|---|
| DNN | 0.953 | 0.735 | 0.481 | 63.6 |
| CNN | 0.887 | 0.240 | 0.095 | 58.7 |
| GRU | 0.872 | 0.234 | 0.109 | 222.3 |
| LSTM | 0.876 | 0.228 | 0.105 | 249.6 |
| Bi-LSTM | 0.872 | 0.230 | 0.106 | 376.5 |

### Key Findings
- DNN achieved the best overall performance across all evaluation metrics.
- Sequential models such as LSTM and GRU performed poorly when combined with TF-IDF representations.
- The DNN model achieved approximately 15.7 reviews per second, making it suitable for real-time deployment.
- Lightweight deep learning architectures can outperform more complex architectures when paired with appropriate feature representations.

---

## Future Improvements
Possible future developments include:
- Using transformer-based embeddings such as BERT or IndoBERT
- Combining TF-IDF with contextual embeddings
- Expanding the dataset to additional e-commerce categories
- Developing a real-time web application for fake review monitoring
- Exploring ensemble and hybrid architectures

---

## Conclusion
This thesis demonstrates the practical application of deep learning for fake review detection in Indonesian e-commerce platforms. The results show that Deep Neural Networks (DNN) provide the best balance between predictive performance and computational efficiency when using TF-IDF feature representations.

The research also highlights the importance of selecting model architectures that align with the underlying data representation. Overall, this project contributes toward the development of scalable and reliable fake review detection systems that can improve trust and transparency in online shopping ecosystems.
