
# Mental health prediction model

This project is a Self-Analysis Mental Health Prediction Model that predicts possible mental health conditions based on user-provided symptoms. The model uses Machine Learning to classify mental health conditions and provides explanations for predictions.



## Features

- Predicts mental health conditions based on symptoms.
- Uses Multi-Class or Multi-Label Classification.
 Preprocessing with StandardScaler & OneHotEncoder.
- Supports multiple ML models (Logistic Regression, Random Forest, XGBoost, etc.).
- Command-Line & Streamlit UI for predictions.




## Structure

📁 Mental_Health_Model 
-  📜 train_model.py # Script for training the model 
-  📜 predict_mental_health.py # CLI script for making predictions 
- 📜 app.py # Streamlit UI for the model 
- 📜 data_cleaning.py # Data preprocessing & cleaning script 
- 📜 requirements.txt # Required Python packages.
- 📜 README.md # Project Documentation
│── 📁 data  
    - mental_health_dataset.csv    
    -  cleaned_mental_health_data.csv  
│── 📁 models  
    - mental_health_model.pkl  
    - scaler.pkl # StandardScaler  
    - condtion_mapping.pkl #For label mappings 


## Model Training Process
##1️⃣ Data Preparation
- Removes unnecessary columns (e.g., ID, Timestamp) 
- Handles missing values (fillna() for categorical & numerical) 
- Encodes categorical variables using OneHotEncoder 
- Scales numerical features using StandardScale.

##2️⃣ Model Training 
- Trains multiple models (Logistic Regression, Random Forest, XGBoost). 
- Selects the best model based on accuracy

##3️⃣ Saving the Model  
- Saves trained model → mental_health_model.pkl.
- Saves label mappings → condition_mapping.pkl
- Saves scaler → scaler.pkl


