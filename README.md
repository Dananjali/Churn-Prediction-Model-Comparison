# Churn-Prediction-Model-Comparison


## 📌 Overview

This project focuses on predicting customer churn using machine learning models based on the **Churn_Modelling.csv** dataset obtained from Kaggle. The dataset simulates banking customer data, and the goal is to identify which customers are likely to leave the bank (churn) based on their features such as credit score, geography, age, balance, and more.

The models built and compared in this project are:
- Random Forest Classifier 🌲
- Logistic Regression 📈
- Support Vector Machine (SVM) 💡
- K-Nearest Neighbors (KNN) 🤝

Dataset used - https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling/data

## 🧠 What I Learned

Throughout this project, I gained hands-on experience in:
- Data preprocessing and cleaning (handling categorical variables, scaling features)
- Label Encoding and One-Hot Encoding
- Visualization using Seaborn and Matplotlib
- Splitting the dataset into training and test sets
- Building and evaluating multiple machine learning models
- Interpreting classification reports and confusion matrices
- Visualizing model performance and comparing them effectively
- Understanding feature importance with ensemble models

## 📊 Exploratory Data Analysis

Before diving into modeling, I performed some analysis to understand the data better.

One specific visualization I included was:
- **Gender vs Churn**: A bar chart showing how many males and females exited the bank. This helped in understanding the gender distribution among churned customers.

I also visualized:
- **Feature Importance** using Random Forest to understand which attributes (e.g., Age, Balance, Geography) had the most influence on customer churn.

## ⚙️ Data Preprocessing

Steps involved:
- Label encoding the **Gender** column (Male/Female → 1/0)
- One-hot encoding the **Geography** column to convert it into machine-readable format
- Selecting relevant features for modeling
- Splitting into train and test sets (80/20 split)
- Feature scaling using `StandardScaler` to normalize numerical data

## 🤖 Model Building & Evaluation

Four different models were trained and evaluated using accuracy, confusion matrix, and classification reports:

| Model                | Accuracy |
|---------------------|----------|
| Random Forest        | ✅ High accuracy and good generalization due to ensemble approach |
| Logistic Regression  | ✅ Simple and interpretable, performs decently |
| Support Vector Machine (SVM) | ✅ Performs well with linear separation |
| K-Nearest Neighbors (KNN)    | ✅ Simpler, but sensitive to feature scaling and less accurate than ensemble methods |

Each model was evaluated using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)
- **Accuracy Score**

I also compared all four models in a **bar chart** showing their accuracy side-by-side for a clear, visual representation of performance.

## ✅ Key Takeaways

- Random Forest performed the best, thanks to its ability to reduce overfitting and capture non-linear relationships.
- Logistic Regression and SVM were also strong contenders, especially for understanding basic decision boundaries.
- KNN struggled slightly in performance but reinforced the importance of feature scaling and parameter tuning.
- Visualizations significantly enhance the interpretability of machine learning results.



## 🙌 Final Thoughts

This project helped me deepen my understanding of practical machine learning workflows — from raw data to deployable insights. It also taught me how to choose, tune, and evaluate different models based on the context and data.

Thanks for reading! Feel free to explore the code and reach out if you'd like to collaborate.

