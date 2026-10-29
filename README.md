# Titanic Survival Prediction

A machine learning classification project predicting passenger survival on the Titanic using the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

## Project overview

The project explores passenger information, preprocesses missing and categorical values, and compares three classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

## Workflow

1. Exploratory data analysis (EDA): survival rates by gender and passenger class, age distributions, and missing-value inspection.
2. Feature preparation: extract passenger titles from names; impute missing values; scale numeric columns; one-hot encode categorical columns.
3. Train/validation split (80/20) with stratification.
4. Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
5. Apply 5-fold cross-validation and GridSearchCV to tune Random Forest and Logistic Regression.
6. Generate predictions for Kaggle's test set and submit a CSV file.

## Results

| Metric | Result |
| --- | --- |
| Logistic Regression validation accuracy | 83.8% |
| Random Forest validation accuracy | 82.12% |
| Decision Tree validation accuracy | 81.6% |
| Tuned Random Forest validation accuracy | 82.68% |
| Kaggle submission score (tuned Logistic Regression) | 0.77272 |

Validation metrics come from the local validation split and are not directly comparable to Kaggle's hidden-label test score.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Google Colab, Kaggle.

## How to run

1. Download `train.csv` and `test.csv` from the Kaggle competition's **Data** tab.
2. Open the script in Google Colab (or adapt the file paths for local execution).
3. Upload the datasets to Colab so they are available as `/content/train.csv` and `/content/test.csv`.
4. Run the training and evaluation code in `titanic.py`.

**Note:** `titanic.py` contains the training and validation analysis. The Kaggle test-set submission code was run separately and is not included in this script.

## Next steps

- Improve feature engineering and validation.
- Compare additional boosting models.
- Add feature-importance analysis and reproducible submission scripts.
