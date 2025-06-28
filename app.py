from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import joblib
from transformers import pipeline
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "your-secret-key"
app.permanent_session_lifetime = timedelta(hours=1)

# Load model and scaler
model = joblib.load("risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load GenAI model
llm = pipeline("text2text-generation", model="google/flan-t5-base")

# Predefined Q&A
faq_data = {
    "what is emi": "EMI stands for Equated Monthly Installment. It's a fixed monthly payment made to repay a loan.",
    "how can i save money": "Create a budget, cut unnecessary expenses, and automate your savings.",
    "what is credit score": "A credit score reflects your ability to repay loans. Higher scores mean better creditworthiness.",
    "how to reduce debt": "Pay more than the minimum amount, avoid taking new loans, and consider debt consolidation.",
    "what is cibil score": "CIBIL score is a 3-digit number that represents your credit history in India. Scores above 750 are considered good.",
    "what is interest rate": "An interest rate is the cost of borrowing money, usually expressed as a percentage of the loan amount.",
    "how to increase credit score": "Pay bills on time, maintain low credit utilization, and avoid frequent loan applications.",
    "should i use credit card for daily expenses": "Only if you can repay the full amount on time. Avoid carrying forward balances.",
    "what is financial planning": "Financial planning involves setting goals and managing your income, expenses, savings, and investments accordingly.",
    "how to create a budget": "List your income and expenses, categorize them, set limits for each, and track spending regularly.",
    "what is emergency fund": "An emergency fund is savings reserved for unexpected expenses like medical bills or job loss—ideally 3–6 months of expenses.",
    "how to escape from credit traps and reduce risk": "To escape credit traps, avoid minimum payments, pay off high-interest debt first, don't take new loans to repay old ones, and track spending. Build an emergency fund to reduce financial risk."
}

model_features = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

@app.route("/")
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "prediction_history" not in session:
        session["prediction_history"] = []
    return render_template("index.html", result=None, advice=None, chat_history=session["chat_history"], csv_results=None, prediction_history=session["prediction_history"])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = {k: float(request.form[k]) for k in model_features}
        df = pd.DataFrame([inputs])

        total_bill = sum(inputs[f"BILL_AMT{i}"] for i in range(1, 7))
        total_paid = sum(inputs[f"PAY_AMT{i}"] for i in range(1, 7))
        avg_bill = total_bill / 6
        payment_gap = total_bill - total_paid

        X_scaled = scaler.transform(df)
        risk_score = model.predict_proba(X_scaled)[0][1]
        prediction = "High Risk" if risk_score >= 0.35 else "Low Risk"

        prompt = f"""
        A user has a risk score of {risk_score:.2f}, with average monthly bills of ₹{avg_bill:.2f}, and a payment gap of ₹{payment_gap:.2f}.
       For more knowledge chat with ai
        """
        advice = llm(prompt, max_new_tokens=100)[0]['generated_text']

        # Save to prediction history
        session.setdefault("prediction_history", []).append({
            "LIMIT_BAL": inputs["LIMIT_BAL"],
            "AGE": inputs["AGE"],
            "Risk Score": round(risk_score, 2),
            "Prediction": prediction
        })
        session.modified = True

        return render_template("index.html", result={
            "Risk Score": round(risk_score, 2),
            "Prediction": prediction
        }, advice=advice, chat_history=session["chat_history"], csv_results=None, prediction_history=session["prediction_history"])

    except Exception as e:
        return render_template("index.html", result={"error": str(e)}, advice=None, chat_history=session["chat_history"], csv_results=None, prediction_history=session.get("prediction_history", []))

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "csvfile" not in request.files:
        return redirect(url_for("index"))

    file = request.files["csvfile"]
    if not file.filename.endswith(".csv"):
        return render_template("index.html", result={"error": "Only CSV files are allowed."}, advice=None, chat_history=session["chat_history"], csv_results=None, prediction_history=session.get("prediction_history", []))

    try:
        df = pd.read_csv(file)
        if not all(f in df.columns for f in model_features):
            missing = list(set(model_features) - set(df.columns))
            return render_template("index.html", result={"error": f"Missing columns in CSV: {missing}"}, advice=None, chat_history=session["chat_history"], csv_results=None, prediction_history=session.get("prediction_history", []))

        X_scaled = scaler.transform(df[model_features])
        scores = model.predict_proba(X_scaled)[:, 1]
        predictions = ["High Risk" if score >= 0.35 else "Low Risk" for score in scores]

        df["Risk Score"] = scores.round(2)
        df["Prediction"] = predictions

        results = df[["LIMIT_BAL", "AGE", "Risk Score", "Prediction"]].to_dict(orient="records")

        session.setdefault("prediction_history", []).extend(results)
        session.modified = True

        return render_template("index.html", result=None, advice=None, chat_history=session["chat_history"], csv_results=results, prediction_history=session["prediction_history"])

    except Exception as e:
        return render_template("index.html", result={"error": str(e)}, advice=None, chat_history=session["chat_history"], csv_results=None, prediction_history=session.get("prediction_history", []))

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    response = faq_data.get(user_message)
    if not response:
        prompt = f"You are a smart financial advisor. Answer this clearly: {user_message}"
        response = llm(prompt, max_new_tokens=100)[0]["generated_text"]

    chat = session.get("chat_history", [])
    chat.append({"user": user_message, "bot": response})
    session["chat_history"] = chat
    session.modified = True

    return jsonify({"response": response})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session.pop("prediction_history", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
