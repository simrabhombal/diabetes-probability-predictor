# 🧠 Diabetes Prediction & LLM-Based Explanation System

This project is an end-to-end **Machine Learning and Explainable AI system** that predicts whether a person has diabetes based on clinical inputs and provides **natural language explanations** using a powerful **LLM (Groq’s LLaMA)**. It combines **predictive modeling**, **model interpretability (SHAP)**, and **interactive UI (Streamlit)**, making it not just a diagnostic tool but also an educational and decision-support system.

---

## 📌 Objective

To build a smart, explainable, and user-friendly system that:
- Predicts the likelihood of a patient having diabetes
- Converts data-driven results into **simple, actionable insights** using a Large Language Model
- Empowers users (patients or healthcare workers) with interpretability and transparency

---

## 📊 Dataset Overview

- **Dataset:** Pima Indians Diabetes Dataset (UCI Repository)
- **Records:** 768
- **Target Variable:** `Outcome` (1 = Diabetic, 0 = Non-Diabetic)
- **Features:**
  - `Pregnancies`
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `Insulin`
  - `BMI`
  - `DiabetesPedigreeFunction`
  - `Age`

---

## 🧪 Machine Learning Workflow

### 1. 🔄 Data Preprocessing
- Removed or imputed missing values
- Feature normalization using `StandardScaler`
- Ensured data balance and integrity

### 2. ⚙️ Model Development
Three models were trained and compared:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

Each model was evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

📌 **Random Forest** was chosen as the final model due to its highest overall performance and robustness.

### 3. 📈 Model Explainability (SHAP)
- SHAP (SHapley Additive exPlanations) was used to:
  - Visualize global feature importance
  - Generate local explanations for individual predictions
  - Highlight which features (e.g., high glucose or BMI) contributed most

### 4. 🔍 Prediction Output
- Final output includes:
  - `Prediction`: Diabetic / Non-Diabetic
  - `Probability Score`: e.g., 71.49%
  - SHAP visualization (bar and force plots)

---

## 🤖 LLM Integration (Groq LLaMA)

To bridge the gap between **technical output and human understanding**, an LLM is used to:
- Analyze the patient’s input values
- Interpret the model’s prediction
- Generate a **natural language summary** including:
  - Health advice
  - Parameter-wise evaluation
  - Confidence tone (reassuring but factual)

📦 Example Output:
> Based on the values you entered, your glucose level of 155 mg/dL is above the normal range, which strongly contributes to a diabetic prediction. Your BMI of 32 also increases the risk. It’s recommended to consult a healthcare provider for further guidance.

---

## 💻 Streamlit Application

An intuitive web-based interface using **Streamlit** that allows:
- Easy input of health parameters
- Immediate results with prediction & probability
- SHAP feature importance plot
- LLM-generated medical reasoning in human-readable format

🎯 **User-friendly** interface for both healthcare professionals and non-technical users.

---

## 🧰 Tech Stack

| Component                | Tool/Library          |
|--------------------------|------------------------|
| Data Handling            | Pandas, NumPy         |
| ML Models                | Scikit-learn, XGBoost |
| Visualization            | Matplotlib, Seaborn   |
| Explainability           | SHAP                  |
| UI                       | Streamlit             |
| Deployment               | Streamlit Cloud       |
| LLM API Integration      | Groq (LLaMA)          |
| API/Secret Management    | dotenv, requests      |

---

## 🛠️ Installation & Usage

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/diabetes-prediction-llm.git
```
### 2. Install required libraries
```bash
pip install -r requirements.txt
```
### 3. Set Up Environment Variable
Create a .env file and enter Groq LLM API
```bash
GROQ_API_KEY=your_groq_key_here
```
### 4. Run streamlit App
Create a .env file and enter Groq LLM API
```bash
streamlit run app.py
```
