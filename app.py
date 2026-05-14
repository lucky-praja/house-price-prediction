from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load("model/model.pkl")
columns = joblib.load("model/columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "area": float(request.form["area"]),
        "bedrooms": int(request.form["bedrooms"]),
        "bathrooms": int(request.form["bathrooms"]),
        "stories": int(request.form["stories"]),
        "parking": int(request.form["parking"]),
        "age": int(request.form["age"]),
        "location": request.form["location"]
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Add missing columns (important for ML model)
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[columns]

    # Prediction
    prediction = model.predict(df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price: ₹{int(prediction)}"
    )

if __name__ == "__main__":
    app.run(debug=True)
