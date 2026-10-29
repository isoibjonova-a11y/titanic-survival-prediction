# Titanic Survival Prediction

A machine-learning classification project using the Kaggle Titanic dataset. The goal is to predict whether a passenger survived the Titanic disaster.

## Project workflow

- Exploratory Data Analysis (EDA): survival by gender and passenger class, age distribution, and missing-value checks.
- Data preprocessing: median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- Model comparison: Logistic Regression, Decision Tree, and Random Forest.
- Evaluation: accuracy, precision, recall, F1-score, confusion matrices, and 5-fold stratified cross-validation.
- Hyperparameter tuning: `GridSearchCV` for Random Forest and Logistic Regression.
- Prediction: generate `submission.csv` for the Kaggle competition using the tuned Logistic Regression model.

## Dataset

Download `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) and place both files in the same folder as `titanic.py`.

The dataset files are not included in this repository.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run

```bash
python titanic.py
```

The script reads `train.csv` and `test.csv` from the current directory and writes `submission.csv` there.

## Results

- Logistic Regression validation accuracy reported in the project: **83.8%**.
- Kaggle submission score recorded for the project: **0.77272**.

Validation accuracy and Kaggle leaderboard score are measured on different data and should not be treated as directly equivalent.

## Models

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## License

MIT


## Kaggle Results

- **Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Kaggle Score:** 0.77272
- **Model:** Optimized Logistic Regression

## Author

- **Kaggle Profile:** [irodasoibjonova](https://www.kaggle.com/irodasoibjonova)
- **GitHub Repository:** Titanic Survival Prediction
