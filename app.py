from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigree"]),
            float(request.form["Age"])
        ]
        features = np.array([data])
        prediction = model.predict(features)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return render_template("result.html", result=result)
    return "Invalid Request"
    
if __name__ == "__main__":
    app.run(debug=True)
