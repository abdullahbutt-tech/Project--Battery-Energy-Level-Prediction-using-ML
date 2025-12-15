from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":

        quantity = int(request.form["quantity"])
        unit_price = float(request.form["unit_price"])
        base_price = float(request.form["base_price"])

        gender = request.form["gender"]
        loyalty_tier = request.form["loyalty_tier"]
        payment_method = request.form["payment_method"]
        region = request.form["region"]
        category = request.form["category"]

        total_price = quantity * unit_price

        if base_price <= 14.67:
            price_tier = "Budget"
        elif base_price <= 23.05:
            price_tier = "Mid"
        elif base_price <= 31.96:
            price_tier = "Premium"
        else:
            price_tier = "Luxury"

        input_df = pd.DataFrame({
            "quantity": [quantity],
            "unit_price": [unit_price],
            "total_price": [total_price],
            "base_price": [base_price],
            "gender": [gender],
            "payment_method": [payment_method],
            "region_x": [region],
            "category": [category],
            "price_tier": [price_tier],
            "loyalty_tier": [loyalty_tier]
        })

        pred = model.predict(input_df)[0]

        if pred == 0:
            prediction = "Cluster 0 – High Value Customers"
        elif pred == 1:
            prediction = "Cluster 1 – Occasional Buyers"
        else:
            prediction = "Cluster 2 – New or Churn Risk Customers"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
