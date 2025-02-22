import joblib
import numpy as np
import pandas as pd

# Load the trained model and preprocessing files
model = joblib.load("mental_health_model.pkl")  # Load trained model
scaler = joblib.load("scaler.pkl")  # Load scaler for numerical data
encoder = joblib.load("encoder.pkl")  # Load encoder for categorical data
condition_mapping = joblib.load("condition_mapping.pkl")  # Load condition label mappings

# Define input symptom list (modify based on your dataset features)
symptom_list = [
    "Growing_Stress", "Changes_Habits", "Mental_Health_History",
    "Mood_Swings", "Coping_Struggles", "Work_Interest", "Social_Weakness",
    "Mental_Health_Interview"
]


def get_user_input():
    """Collect user symptoms from command-line input."""
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
    """Make predictions based on user symptoms."""
    user_data = get_user_input()

    # Preprocess input (scale & encode)
    user_data_scaled = scaler.transform(user_data)

    # Predict mental health condition
    prediction = model.predict(user_data_scaled)[0]
    predicted_condition = condition_mapping[prediction]

    print(f"\n🧠 Predicted Mental Health Condition: **{predicted_condition}**")


if __name__ == "__main__":
    predict_mental_health()