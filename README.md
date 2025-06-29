# 💸 FinSafeAI - Risk & Coaching Assistant

FinSafeAI is a smart, AI-powered platform that predicts a user's credit risk and provides personalized financial advice using GenAI. It supports **voice input**, **CSV upload for bulk predictions**, and **real-time AI chatbot interaction** — making it inclusive, interactive, and insightful.

---

## 🚀 Features

### 🧾 Risk Prediction
- Predicts user's credit risk using a pre-trained ML model.
- Displays both **Risk Score** and a clear **Low/High Risk** label.

### 🎤 Voice Input Support
- Users can **speak values** directly into the form fields.
- Built using `webkitSpeechRecognition` and `words-to-numbers`.

### 📂 Bulk CSV Upload
- Upload a CSV file to get predictions for **multiple users** at once.
- Outputs include LIMIT_BAL, AGE, Risk Score, and Prediction.

### 🤖 GenAI Financial Coaching
- Integrated with **Flan-T5** to provide personalized financial advice.
- Based on user’s predicted risk, average bills, and payment gap.

### 💬 Chatbot Integration
- Session-based chatbot helps users understand financial concepts.
- Includes FAQs + real-time GPT-powered responses.

### 🧠 Prediction History (Session Based)
- Remembers past predictions and chatbot messages during the session.

---

## 🗂 Folder Structure

```
FinSafeAI/
│
├── static/                     # (Optional) For JS, CSS, etc.
├── templates/
│   └── index.html              # Frontend template
├── risk_model.pkl              # Trained ML model
├── scaler.pkl                  # Scaler for preprocessing
├── app.py                      # Flask backend
├── requirements.txt            # Python dependencies
└── README.md                   # Project README
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Niraj-Patel-1901/FinSafeAI.git
   cd FinSafeAI
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   python app.py
   ```

5. Open browser and visit `http://127.0.0.1:5000`

---

## 📊 Sample CSV Format

Your CSV must contain **exactly** the following columns:

```
LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
```

---

## 📦 Requirements

Save this as `requirements.txt`:

```
Flask
pandas
joblib
transformers
torch
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Details

- Model: `RandomForestClassifier`
- Dataset: Based on the **UCI Credit Card Dataset** (cleaned & preprocessed)
- Input: 23 financial features
- Output: Credit risk score + classification

---

## ✨ Acknowledgements


- Uses **Google Flan-T5** model for GenAI responses
- Special thanks to mentors and organizers for the opportunity

---

## 🏁 Status

✅ Successfully built and tested locally  
✅ Passed first evaluation round  
🚀 Ready for demo & deployment

---

## 📧 Contact

**Niraj Patel**  
GitHub: github.com/Niraj-Patel-1901  
Email: nirajpatel49@apsit.edu.in

---

> 💡 _“Making finance safe, simple, and smart — for everyone.”_
