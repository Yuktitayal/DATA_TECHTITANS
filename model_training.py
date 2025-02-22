import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("cleaned_mental_health_data1.csv")
condition = "Mood_Swings"

label_encoder = LabelEncoder()
df[condition] = label_encoder.fit_transform(df[condition])

X = df.drop(columns=[condition])
y = df[condition]
df[condition] = label_encoder.fit_transform(df[condition])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
top_features_idx = np.argsort(importances)[-8:]
top_feature_names = X.columns[top_features_idx]

X_train_selected = X_train[:, top_features_idx]
X_test_selected = X_test[:, top_features_idx]


def train_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))

    return accuracy_score(y_test, y_pred), model


logreg_acc, logreg_model = train_model(LogisticRegression(), "Logistic Regression")
rf_acc, rf_model = train_model(RandomForestClassifier(n_estimators=100), "Random Forest")
xgb_acc, xgb_model = train_model(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), "XGBoost")

models = {"Logistic Regression": logreg_acc, "Random Forest": rf_acc, "XGBoost": xgb_acc}
best_model_name = max(models, key=models.get)

if best_model_name == "Logistic Regression":
    best_model = logreg_model
elif best_model_name == "Random Forest":
    best_model = rf_model
else:
    best_model = xgb_model

joblib.dump(best_model, "mental_health_model.pkl")
print(f"\nBest Model: {best_model_name} saved as 'mental_health_model.pkl'")
joblib.dump(scaler, "scaler.pkl")
print(" e scaler saved as'scaler.pkl'")
condition_mapping = dict(enumerate(label_encoder.classes_))
joblib.dump(condition_mapping, "condition_mapping.pkl")
print("condition_mapping saved as 'condition_mapping.pkl'")
