from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("logistic_model_loan.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get form data
        features = [
            int(request.form['Gender']),
            int(request.form['Married']),
            int(request.form['Education']),
            int(request.form['Self_Employed']),
            int(request.form['Dependents']),
            np.log(float(request.form['ApplicantIncome'])),
            np.log1p(float(request.form['CoapplicantIncome'])),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            int(request.form['Credit_History']),
            int(request.form['Property_Area'])
        ]

        prediction = model.predict([features])[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
