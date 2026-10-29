# -*- coding: utf-8 -*-
"""Titanic Survival Prediction: EDA, model comparison, tuning, and Kaggle submission."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold



sns.set_theme(style = "whitegrid")

print("Titanic loyihasi!")

df = pd.read_csv("train.csv")
df.head()

df.shape

df.info()

df.isnull().sum()

print("Ustunlar:")
print(df.columns.tolist())

print("\nSurvived qiymatlari:")
print(df["Survived"].value_counts())

print("\nSurvived foizlari:")
print(df["Survived"].value_counts(normalize=True) * 100)

sns.barplot(data=df, x="Sex", y="Survived")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

sns.barplot(data=df, x="Pclass", y="Survived")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

sns.histplot(
    data=df,
    x="Age",
    hue="Survived",
    bins=30,
    kde=True,
    multiple="layer"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percent": (missing / len(df) * 100).round(2)
})

missing_df = missing_df[missing_df["Missing_Count"] > 0]
missing_df.sort_values("Missing_Percent", ascending=False)

print("Age median:", df["Age"].median())
print("\nEmbarked value counts:")
print(df["Embarked"].value_counts())

df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")

print(df["Title"].value_counts())

df["Title"] = df["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
     "Major", "Rev", "Sir", "Jonkheer", "Dona"],
    "Rare"
)


df["Title"] = df["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})

print(df["Title"].value_counts())

X = df.drop(columns=["Survived"])
y = df["Survived"]

print("Belgilar o'lchami:", X.shape)
print("Maqsadli ustun o'lchami:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("O'qitish ma'lumotlari:", X_train.shape)
print("Tekshirish ma'lumotlari:", X_val.shape)

numeric_features = ["Age", "SibSp", "Parch", "Fare"]

categorical_features = ["Pclass", "Sex", "Embarked" , "Title"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_val)


accuracy = accuracy_score(y_val, y_pred)
print("Model aniqligi:", round(accuracy, 4))
print(classification_report(y_val, y_pred))

"""Logistic Regression modeli validation ma’lumotlarida **83.8% aniqlikka (accuracy)** erishdi. Omon qolmagan yo‘lovchilar uchun F1-score 0.87, omon qolgan yo‘lovchilar uchun esa 0.78 ni tashkil etdi"""

cm = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Omon qolmagan", "Omon qolgan"]
)

disp.plot(cmap="Blues")
plt.title("Logistic Regression — Confusion Matrix")
plt.show()

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_val)

print("Decision Tree aniqligi:", round(
    accuracy_score(y_val, tree_pred), 4
))

print("\nModelning batafsil hisoboti:")
print(classification_report(y_val, tree_pred))

"""Logistic Regression modeli Decision Tree modelidan yuqoriroq Accuracy va F1-score ko‘rsatdi."""

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    ))
])


rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_val)

print("Random Forest aniqligi:",
      round(accuracy_score(y_val, rf_pred), 4))

print("\nModelning batafsil hisoboti:")
print(classification_report(y_val, rf_pred))

"""Random Forest modeli validation ma’lumotlarida **82.12% aniqlikka (accuracy)** erishdi. Omon qolmagan yo‘lovchilar uchun F1-score 0.86, omon qolgan yo‘lovchilar uchun esa 0.75 ni tashkil etdi. Logistic Regression modeli 83.8% aniqlik va omon qolganlar uchun 0.78 F1-score bilan yaxshiroq natija ko‘rsatdi."""

models = {
    "Logistic Regression": (model, y_pred),
    "Decision Tree": (tree_model, tree_pred),
    "Random Forest": (rf_model, rf_pred)
}

results = []


for name, (estimator, predictions) in models.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions),
        "Recall": recall_score(y_val, predictions),
        "F1-score": f1_score(y_val, predictions)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1-score", ascending=False
).reset_index(drop=True)

results_df.round(3)

"""Uchta klassifikatsiya modeli taqqoslandi. **Logistic Regression 83.8% Accuracy bilan eng yaxshi natijani ko‘rsatdi.** Shuningdek, omon qolgan yo‘lovchilarni aniqlashda 0.739 Recall va 0.779 F1-score bilan boshqa modellardan ustun bo‘ldi. Random Forest 82.1%, Decision Tree esa 81.6% Accuracy natijasiga erishdi. Natijalarga ko‘ra, ushbu uchta model orasida Logistic Regression eng yaxshi natija ko‘rsatdi."""

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


models = {
    "Logistic Regression": model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model
}

for name, estimator in models.items():
    scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring="f1"
    )

    print(f"\n{name}")
    print("Har bir fold natijasi:", scores.round(3))
    print("O'rtacha F1-score:", round(scores.mean(), 3))
    print("Standart og'ish:", round(scores.std(), 3))

from sklearn.model_selection import GridSearchCV


param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}


grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)


print("Eng yaxshi parametrlar:", grid_search.best_params_)
print("Eng yaxshi CV F1-score:", round(grid_search.best_score_, 3))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
best_model = grid_search.best_estimator_

best_pred = best_model.predict(X_val)

print("Accuracy:", round(accuracy_score(y_val, best_pred), 4))

print("\nModelning batafsil hisoboti:")
print(classification_report(y_val, best_pred))

cm = confusion_matrix(y_val, best_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Omon qolmagan", "Omon qolgan"]
)

disp.plot(cmap="Blues")
plt.title("Optimallashtirilgan Random Forest")
plt.show()

"""Optimallashtirilgan Random Forest modelini baholash

GridSearchCV yordamida optimallashtirilgan Random Forest modeli validation ma’lumotlarida 82.68% Accuracy natijasini ko‘rsatdi.

Omon qolmagan yo‘lovchilar (0-sinf) uchun F1-score 0.87, omon qolgan yo‘lovchilar (1-sinf) uchun esa 0.76 ni tashkil etdi. Omon qolgan yo‘lovchilar uchun Recall 0.70 bo‘ldi, ya’ni model haqiqatan omon qolgan yo‘lovchilarning 70 foizini to‘g‘ri aniqladi.

Cross-Validation natijasida modelning o‘rtacha F1-score qiymati 0.749 dan 0.759 gacha yaxshilandi. Validation natijasi esa 0.8268 Accuracy va 0.76 F1-score bilan yakunlandi.
"""

param_grid_lr = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"],
    "classifier__max_iter": [1000, 2000]
}

grid_search_lr = GridSearchCV(
    estimator=model,
    param_grid=param_grid_lr,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search_lr.fit(X_train, y_train)

print("Eng yaxshi parametrlar:", grid_search_lr.best_params_)
print("Eng yaxshi CV F1-score:", round(grid_search_lr.best_score_, 3))

best_lr_model = grid_search_lr.best_estimator_

lr_pred = best_lr_model.predict(X_val)

print("Accuracy:", round(accuracy_score(y_val, lr_pred), 4))

print("\nModelning batafsil hisoboti:")
print(classification_report(y_val, lr_pred))

cm_lr = confusion_matrix(y_val, lr_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Omon qolmagan", "Omon qolgan"]
)

disp.plot(cmap="Blues")
plt.title("Optimallashtirilgan Logistic Regression")
plt.show()

"""### Optimallashtirilgan Logistic Regression modelini baholash

`GridSearchCV` yordamida tanlangan Logistic Regression modeli validation ma’lumotlarida **83.8% Accuracy** natijasiga erishdi.

Omon qolmagan yo‘lovchilar uchun F1-score 0.87, omon qolgan yo‘lovchilar uchun esa 0.78 ni tashkil etdi. Omon qolgan yo‘lovchilar uchun Recall 0.74 bo‘ldi, ya’ni model haqiqatan omon qolgan yo‘lovchilarning 74 foizini to‘g‘ri aniqladi.

Optimallashtirilgan modelning validation natijalari dastlabki Logistic Regression modeli bilan bir xil bo‘ldi. Cross-Validation bo‘yicha o‘rtacha F1-score ham 0.759 ni tashkil etdi.

"""

test_df = pd.read_csv("test.csv")

print("Test ma'lumotlari o'lchami:", test_df.shape)
print(test_df.head())


print("\nBo'sh qiymatlar:")
print(test_df.isnull().sum())


test_df["Title"] = test_df["Name"].str.extract(r",\s*([^.]*)\.")


test_df["Title"] = test_df["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
     "Major", "Rev", "Sir", "Jonkheer", "Dona"],
    "Rare"
)


test_df["Title"] = test_df["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})


X_test = test_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])


test_predictions = best_lr_model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions
})


submission.to_csv("submission.csv", index=False)

print("\nBashorat natijalari:")
print(submission.head(10))

print("\nBashorat qilingan omon qolganlar soni:",
      (test_predictions == 1).sum())

print("Bashorat qilingan omon qolmaganlar soni:",
      (test_predictions == 0).sum())

print("\nsubmission.csv fayli tayyor!")

print("Saved submission file: submission.csv")

