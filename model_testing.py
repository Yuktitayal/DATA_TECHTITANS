
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\Dataset.csv")


columns_to_remove = ["Timestamp","Gender","Country","Occupation","self_employed","family_history","treatment","Days_Indoor","care_options"]  # Replace with actual column names to drop
df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])  # Drop only existing columns

imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


label_encoders = {}
for col in df_imputed.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col])
    label_encoders[col] = le


plt.figure(figsize=(10, 5))
sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


df_imputed.to_csv('cleaned_mental_health_data1.csv', index=False)
print("Preprocessing complete. Cleaned data saved as 'cleaned_mental_health_data1.csv'.")





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
condition= "Mood_Swings"

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
joblib.dump(scaler,"scaler.pkl")
print(" e scaler saved as'scaler.pkl'")
condition_mapping = dict(enumerate(label_encoder.classes_))
joblib.dump(condition_mapping,"condition_mapping.pkl")
print("condition_mapping saved as 'condition_mapping.pkl'")



import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load("mental_health_model.pkl")


scaler = joblib.load("scaler.pkl")


symptom_list = [
    "Growing_Stress","Changes_Habits","Mental_Health_History","Mood_Swings","Coping_Struggels","Work_Interest","Social_Weaknes","mental_health_interview"
]

def get_user_input():
    """Collect user symptoms and convert to numerical format"""
    print("\nEnter your symptoms (0 for No, 1 for Yes):")
    user_input = []
    for symptom in symptom_list:
        while True:
            try:
                val = int(input(f"Do you experience {symptom.replace('_', ' ')}? (0/1): "))
                if val in [0, 1]:
                    user_input.append(val)
                    break
                else:
                    print("Invalid input! Enter 0 (No) or 1 (Yes).")
            except ValueError:
                print("Invalid input! Enter a number.")

    return np.array(user_input).reshape(1, -1)

def predict_mental_health():
    """Make predictions based on user symptoms"""
    user_data = get_user_input()
    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)[0]
    

    condition_mapping = joblib.load("condition_mapping.pkl")
    predicted_condition = condition_mapping[prediction]

    print(f"\n🧠 Predicted Mental Health Condition: {predicted_condition}")

if __name__ == "__main__":
    predict_mental_health()







