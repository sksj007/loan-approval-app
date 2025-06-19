# 🏦 Loan Approval App

A simple machine learning web app that predicts whether a loan will be approved based on user input.

## 🔧 Features

- Web form to enter applicant details
- Predicts loan approval in real-time
- Trained ML model using historical data
- Built with Python, Flask, scikit-learn

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/sksj007/loan-approval-app.git
   cd loan-approval-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python model/train_model.py
   ```

4. Run the app:
   ```bash
   python app.py
   ```
   Then go to `http://127.0.0.1:5000/` in your browser.

## 📁 Project Structure

```
loan-approval-app/
├── app.py               # Flask app
├── model/train_model.py # Model training script
├── model/model.pkl      # Saved ML model
├── templates/index.html # Web page
├── requirements.txt     # Python dependencies
```

## 📊 ML Model

- Algorithm: Logistic Regression (you can change it)
- Preprocessing: Handled missing values, label encoding
- Input features: Gender, Income, Loan Amount, Credit History, etc.

## 📌 Notes

- Based on the popular Loan Prediction dataset (e.g., from Kaggle)
- Easily customizable for other datasets or ML models

---

Let me know if you’d like help deploying this app online (Heroku, Render, etc.)
