# 🧠 Multi-Layer Perceptron Classifiers on Benchmark Datasets

This repository demonstrates hands-on use of **neural networks (Multi-Layer Perceptrons - MLPs)** for multi-class classification on classic tabular datasets. All code is carefully documented and structured, making it ideal for recruiters and technical specialists to evaluate deep learning, data preprocessing, and pipeline engineering skills in a **real-world, reproducible context**.

---

## 🚀 Project Focus

- **Apply MLPs to structured, multivariate datasets with categorical targets**
- Automate all steps: data ingestion, categorical encoding, scaling, training, and evaluation
- Showcase *practical challenges* (such as unbalanced classes and symbolic features)
- Present **critical analysis** and model limitations, demonstrating data science maturity

---

## 📄 Datasets Used

| Dataset                                     | Classes | Description                                                    | Source                                                                                 |
|----------------------------------------------|---------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Obesity Estimation**                      | 7       | Predict obesity category from lifestyle and biometric data      | [UCI 544](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) |
| **Car Evaluation**                          | 4       | Assess car acceptability from technical attributes              | [UCI 19](https://archive.ics.uci.edu/dataset/19/car+evaluation)                        |
| **Contact Lenses Recommendation**           | 3       | Recommend lens type from basic patient information              | [UCI 58](https://archive.ics.uci.edu/dataset/58/lenses)                                |

All datasets are loaded programmatically using [`ucimlrepo`](https://github.com/RUB-SysSec/ucimlrepo), requiring **zero manual setup**.

---

## 🏗️ Pipeline: Core Steps

For each dataset, the notebook performs:

1. **Dataset Import & Metadata Display**
   - Fetches data and variables
   - Prints descriptions and schema for transparency

2. **Preprocessing & Feature Engineering**
   - Encodes categorical/binary variables (`pd.get_dummies` for features, `LabelEncoder` for targets)
   - Splits data (stratified `train_test_split`)
   - Standardizes with `StandardScaler` for numeric stability

3. **Modeling**
   - **MLP design:** Typically 2 hidden layers (ReLU), units tailored per dataset
   - Output layer: *Softmax* activation, multiclass
   - Loss: `sparse_categorical_crossentropy`

4. **Training**
   - Hyperparameters: epochs, batch sizes, validation split, silent logs
   - Explores effect of epochs and architecture for robust results

5. **Evaluation**
   - Outputs **Accuracy**, **Precision**, **Recall**, **F1 Score** (weighted)
   - Compares predictions with ground-truth, prints final performance
   - Notes on dataset difficulties, model performance, overfitting, and edge-cases

---

## 📊 Results Summary

| Dataset           | Accuracy | Precision | Recall | F1-score | Comments |
|-------------------|----------|-----------|--------|----------|-------------------------------------------------------------|
| Obesity           | 26%      | 26%       | 26%    | 24%      | Extreme class imbalance & high number of classes             |
| Car Evaluation    | 27%      | 24%       | 27%    | 24%      | High symbolic complexity; rules not easily learned by MLP    |
| Lenses            | 59%      | 35%       | 59%    | 44%      | Very small & rule-based dataset; MLP unable to generalize    |

> **Insight:** While deep neural networks are powerful, highly categorical or rule-based tabular datasets often need tailored architectures, specialized encoders, or different algorithms entirely. This project includes observations on these practical limits—essential for real data science work.

---

## 💡 Features & Strengths

- **Full Automation**: All steps (from fetch to final metrics) are automated and reproducible.
- **Production-Ready Structure**: Clean module separation—swap data/models easily.
- **Critical Analysis**: Each section includes commentary about trade-offs and pitfalls for nuanced evaluation.
- **Didactic Value**: Useful for both professionals and learners; easily extensible for interviews, tests, or further research.

---

## 🛠️ Technologies

- **Python 3.10+**
- **TensorFlow (Keras)**
- **scikit-learn**
- **pandas**
- **ucimlrepo**

---

## 💻 Example (Obesity Dataset)

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

Fetch and preprocess data
data = fetch_ucirepo(id=544)
X = pd.get_dummies(data.data.features, drop_first=True)
y = LabelEncoder().fit_transform(data.data.targets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Model
model = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape,)),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(len(set(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=75, batch_size=16, validation_split=0.2, verbose=0)

Evaluation
y_pred = model.predict(X_test).argmax(axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

text

---

## 📌 Usage

1. Install dependencies:
    ```
    pip install ucimlrepo tensorflow scikit-learn pandas
    ```
2. Run `Multi-Layer-Perceptron.ipynb` in Jupyter or Colab (recommended).

*No config or manual data downloads required!*

---

## 🎯 Recruiter & Specialist Appeal

- **Clear, modular code** — focus on maintainability and readability.
- **Honest reflection on technical chiallenges** — shows critical thinking and realistic expectations.
- **Solid foundations for industry work** — pipelines, evaluation, and documentation mirror professional practice.
- **Open for extension** — ideal as a baseline for technical interviews, case studies, or sandbox experiments.

---

## 👨‍💻 Author

**Thiago Aragão**  
Data Scientist | Deep Learning | Generative AI | NLP | Computer Vision
- GitHub: [@DrAragorn](https://github.com/DrAragorn)
- Email: thiago.alpha.06@gmail.com
- LinkedIn: [linkedin.com/in/thiago-r-aragao](https://linkedin.com/in/thiago-r-aragao)

---

*This repository is a showcase of practical neural network application, robust pipeline building, and critical evaluation—ready for exploration by hiring managers and technical peers.*
