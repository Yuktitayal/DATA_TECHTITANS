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
