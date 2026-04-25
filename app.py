from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model/car_model.pkl', 'rb'))
columns = pickle.load(open('model/columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    present_price = float(data['Present_Price'])
    kms = float(data['Kms_Driven'])
    owner = int(data['Owner'])
    year = int(data['model_year'])

    car_age = 2025 - year
    km_per_year = kms / (car_age + 1)

    fuel = data['fuel_type']
    seller = data['Seller_Type']
    transmission = data['transmission']
    brand = data['brand']

    input_data = [0] * len(columns)

    for i, col in enumerate(columns):
        if col == 'Present_Price':
            input_data[i] = present_price
        elif col == 'Kms_Driven':
            input_data[i] = kms
        elif col == 'Owner':
            input_data[i] = owner
        elif col == 'car_age':
            input_data[i] = car_age
        elif col == 'km_per_year':
            input_data[i] = km_per_year
        elif col == f"Fuel_Type_{fuel}":
            input_data[i] = 1
        elif col == f"Seller_Type_{seller}":
            input_data[i] = 1
        elif col == f"Transmission_{transmission}":
            input_data[i] = 1
        elif col == f"brand_{brand}":
            input_data[i] = 1

    input_df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(input_df)

    return render_template('index.html',
        prediction_text=f"Predicted Price: ₹ {round(prediction[0],2)} lakhs")

if __name__ == "__main__":
    app.run(debug=True)