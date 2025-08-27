# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn (leave the service) based on demographic, service usage, and account details.  
The project leverages the **Telco Customer Churn dataset**, applies preprocessing and feature engineering, and trains a Random Forest Classifier to generate churn predictions.

---

## Features

- Preprocess customer data (handle missing values, encode categorical features, scale numerical features).
- Perform Exploratory Data Analysis (EDA) with churn trend visualizations.
- Train a machine learning model (Random Forest Classifier) for churn prediction.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Save and reuse trained models (`rf_model.pkl`) and feature sets (`model_features.pkl`).
- Run Jupyter Notebook for experimentation or Python script for quick predictions.
- Extendable for deployment as a web app (Flask/Streamlit).

---

## Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, joblib  
- **Model:** Random Forest Classifier  
- **Environment:** Jupyter Notebook & Python scripts  

---

## Installation & Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. **Create and activate virtual environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate       # Linux/Mac
    venv\Scripts\activate          # Windows
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

- **Run Jupyter Notebook** for analysis and training:

    ```bash
    jupyter notebook notebooks/Customer\ Churn\ Prediction.ipynb
    ```

- **Run Python script** for predictions:

    ```bash
    python src/Customer_Churn_Prediction.py
    ```

- **Dataset:**  
  Located in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

- **Outputs:**  
  - Trained model → `models/rf_model.pkl`  
  - Features list → `models/model_features.pkl`  

---

## Future Improvements

- Experiment with other ML algorithms (Logistic Regression, XGBoost, Neural Networks).  
- Perform hyperparameter tuning with GridSearchCV/RandomizedSearchCV.  
- Deploy as a Flask/Streamlit app for real-time predictions.  
- Automate retraining with new incoming data.  
- Build an interactive dashboard for churn analytics.  

---

## Acknowledgments

- [Telco Customer Churn Dataset (IBM)](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)  
- [scikit-learn](https://scikit-learn.org/)  
- [pandas](https://pandas.pydata.org/)  
- [seaborn](https://seaborn.pydata.org/)  

---

## Contact

For questions or feedback, please contact:  
**Adi Karthikeya S B** – [adikarthikeya1234@gmail.com]  
GitHub: [https://github.com/adikarthikeya2003](https://github.com/adikarthikeya2003)  
