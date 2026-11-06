import os
import pickle
import warnings

import pandas as pd
from flask import Flask, jsonify, request, render_template

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_USD_INR = 88.0  # editable on the page; update to the current rate

# Same column order the model was trained with
FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "city", "country",
]

app = Flask(__name__)

with open(os.path.join(BASE, "houseprice.pkl"), "rb") as f:
    model = pickle.load(f)


def load_cities():
    """LabelEncoder codes cities alphabetically, so the index in the sorted
    unique list is exactly the code the model was trained with."""
    try:
        df = pd.read_csv(os.path.join(BASE, "data.csv"))
        return sorted(df["city"].dropna().unique().tolist())
    except Exception:
        return []


CITIES = load_cities()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/cities")
def cities():
    return jsonify([{"code": i, "name": c} for i, c in enumerate(CITIES)])


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        rate = float(data.get("rate") or DEFAULT_USD_INR)
        if rate <= 0:
            return jsonify({"error": "Exchange rate must be above 0."}), 400

        row = {k: float(data[k]) for k in FEATURES if k != "country"}
        row["country"] = 1  # USA, same mapping as main.py
        x = pd.DataFrame([row], columns=FEATURES)

        usd = float(model.predict(x)[0])

        # Per-feature contribution = coefficient x value (linear model)
        parts = [{"name": "Base value (intercept)", "value": "",
                  "usd": float(model.intercept_),
                  "inr": float(model.intercept_) * rate}]
        for name, coef in zip(FEATURES, model.coef_):
            c = float(coef) * row[name]
            parts.append({"name": name, "value": row[name],
                          "usd": c, "inr": c * rate})

        return jsonify({
            "usd": usd,
            "inr": usd * rate,
            "rate": rate,
            "parts": parts,
            "warning": "The model returned a price at or below zero for these "
                       "inputs. Check the values." if usd <= 0 else None,
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Every field must be a valid number."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
