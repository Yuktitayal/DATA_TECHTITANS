# DATA_TECHTITANS
Self analysis of mental health model
**Mental Health Prediction Model**
📌 Overview
This project is a Self-Analysis Mental Health Prediction Model that predicts possible mental health conditions based on user-provided symptoms. The model uses Machine Learning to classify mental health conditions and provides explanations for predictions.

🚀 Features
✔ Predicts mental health conditions based on symptoms.
✔ Uses Multi-Class or Multi-Label Classification.
✔ Preprocessing with StandardScaler & OneHotEncoder.
✔ Supports multiple ML models (Logistic Regression, Random Forest, XGBoost, etc.).
✔ Command-Line & Streamlit UI for predictions.
✔ SHAP/LIME for Explainability (Optional).

📂 Project Structure
graphql
Copy
Edit
📁 Mental_Health_Model
│── 📜 train_model.py          # Script for training the model
│── 📜 predict_mental_health.py # CLI script for making predictions
│── 📜 app.py                  # Streamlit UI for the model
│── 📜 data_cleaning.py        # Data preprocessing & cleaning script
│── 📜 requirements.txt        # Required Python packages
│── 📜 README.md               # Project Documentation
│── 📁 data
│   ├── mental_health_dataset.csv  # Raw dataset
│   ├── cleaned_mental_health_data.csv # Processed dataset
│── 📁 models
│   ├── mental_health_model.pkl # Trained ML model
│   ├── scaler.pkl              # StandardScaler for preprocessing
│   ├── encoder.pkl             # OneHotEncoder for categorical features
│   ├── condition_mapping.pkl    # Label mapping for conditions
🔧 Installation & Setup
1️⃣ Clone the Repository
sh
Copy
Edit
git clone https://github.com/yourusername/Mental_Health_Model.git
cd Mental_Health_Model
2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Model
📌 Option 1: Command-Line Interface (CLI)
sh
Copy
Edit
python predict_mental_health.py
👉 Follow the on-screen prompts and input symptoms (0 for No, 1 for Yes).

📌 Option 2: Web-Based UI (Streamlit)
sh
Copy
Edit
streamlit run app.py
👉 This will open a user-friendly web app for predictions.

🛠 Model Training Process
📌 1️⃣ Data Preparation
Removes unnecessary columns (e.g., ID, Timestamp)
Handles missing values (fillna() for categorical & numerical)
Encodes categorical variables using OneHotEncoder
Scales numerical features using StandardScaler
📌 2️⃣ Model Training
Trains multiple models (Logistic Regression, Random Forest, XGBoost).
Selects the best model based on accuracy.
📌 3️⃣ Saving the Model & Preprocessors
Saves trained model → mental_health_model.pkl
Saves scaler & encoder → scaler.pkl, encoder.pkl
Saves label mappings → condition_mapping.pkl
📊 Model Performance (Example Results)
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	85.2%	83.1%	81.5%	82.3%
Random Forest	87.5%	86.2%	84.7%	85.4%
XGBoost	89.1%	88.4%	87.2%	87.8%
👉 XGBoost performs the best and is used as the final model.
