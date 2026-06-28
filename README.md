# Iris-Flower-Classification

# 🌸 Iris Flower Classification using Machine Learning

> Predicting Iris flower species using Machine Learning with feature engineering, model comparison, hyperparameter tuning, and insightful visualizations.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Overview

This project builds and evaluates multiple Machine Learning models to classify Iris flowers into one of three species:

- 🌼 Setosa
- 🌸 Versicolor
- 🌺 Virginica

Using the famous Iris dataset, the project includes:

- 🌿 Feature Engineering
- 🤖 Training 8 Machine Learning models
- 🔍 Hyperparameter Tuning
- 📊 Model Evaluation
- 📈 Data Visualization
- 🏆 Best Model Selection

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | UCI Machine Learning Repository / Scikit-Learn |
| **Samples** | 150 |
| **Classes** | 3 |
| **Distribution** | Balanced (50 samples each) |
| **Features** | Sepal Length, Sepal Width, Petal Length, Petal Width |

---

## 📁 Project Structure

```text
IrisClassification/
│
├── data/
│   └── iris.csv
│
├── plots/
│   ├── 01_pairplot.png
│   ├── 02_boxplots.png
│   ├── 03_correlation.png
│   ├── 04_model_comparison.png
│   ├── 05_confusion_matrix.png
│   ├── 06_feature_importance.png
│   ├── 07_decision_boundary.png
│   └── 08_learning_curve.png
│
├── src/
│   └── iris_classification.py
│
├── results_summary.txt
├── requirements.txt
└── README.md
```

---

# 🌿 Feature Engineering

Seven additional features were created from the original dataset.

| Feature | Formula |
|---------|---------|
| Petal Ratio | Petal Length / Petal Width |
| Sepal Ratio | Sepal Length / Sepal Width |
| Petal Area | Petal Length × Petal Width |
| Sepal Area | Sepal Length × Sepal Width |
| Length Difference | Sepal Length − Petal Length |
| Width Difference | Sepal Width − Petal Width |
| Petal/Sepal Area Ratio | Petal Area / Sepal Area |

---

# 🤖 Machine Learning Models

The following algorithms were trained and compared.

| Model | CV Accuracy | Test Accuracy | ROC-AUC |
|------|:-----------:|:-------------:|:------:|
| Logistic Regression | 96.67% | **100%** | **1.0000** |
| K-Nearest Neighbors | 96.67% | **100%** | **1.0000** |
| Support Vector Machine | 97.50% | **100%** | **1.0000** |
| Decision Tree | 94.17% | 86.67% | 0.9000 |
| Random Forest | 96.67% | 96.67% | 0.9933 |
| Extra Trees | 97.50% | 96.67% | 0.9967 |
| Gradient Boosting | 95.00% | 83.33% | 0.9800 |
| Naive Bayes | 96.67% | 96.67% | 1.0000 |

---

# 🏆 Best Model

**Logistic Regression**

| Metric | Score |
|---------|------|
| Test Accuracy | **100%** |
| ROC-AUC | **1.0000** |
| Cross Validation Accuracy | **96.67%** |

### Hyperparameter Tuning

```
GridSearchCV (5-Fold Cross Validation)

Best Parameters:
C = 1
solver = lbfgs
```

---

# 📈 Visualizations

The project automatically generates the following visualizations.

| Visualization | Description |
|--------------|-------------|
| 📊 Pair Plot | Feature relationships |
| 📦 Box Plots | Distribution of features |
| 🔥 Correlation Heatmap | Feature correlation |
| 📉 Model Comparison | Accuracy & ROC comparison |
| 🎯 Confusion Matrix | Best model predictions |
| 🌿 Feature Importance | Random Forest importance |
| ✨ Decision Boundary | SVM visualization |
| 📈 Learning Curve | Model performance over training |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/IrisClassification.git
```

Move into the project directory

```bash
cd IrisClassification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python src/iris_classification.py
```

---

# 📊 Results

✅ Compared **8 Machine Learning Algorithms**

✅ Performed **Feature Engineering**

✅ Applied **GridSearchCV Hyperparameter Tuning**

✅ Generated **8 Visualizations**

✅ Achieved **100% Test Accuracy**

✅ ROC-AUC Score **1.0000**

---

# 💡 Key Findings

- 🌸 Petal features are the strongest predictors of Iris species.
- 🌿 Setosa is perfectly separable from the other classes.
- 📊 Engineered features improved the performance of linear models.
- 🤖 Logistic Regression and SVM achieved perfect classification on the test set.
- 📈 Petal Length and Petal Width contributed the most to prediction accuracy.

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

# 📌 Future Improvements

- Deploy with Streamlit
- Add a web interface for predictions
- Experiment with Deep Learning
- Implement automated feature selection
- Build an interactive dashboard

---

## ⭐ If you found this project helpful, consider giving it a Star on GitHub!

---
