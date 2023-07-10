from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('random_forest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    present_price = float(request.form.get('present_price'))
    kms_driven = int(request.form.get('kms_driven'))
    past_owners = int(request.form.get('past_owners'))
    age = int(request.form.get('age'))
    fuel_type_diesel = bool(request.form.get('fuel_type_diesel'))
    fuel_type_petrol = bool(request.form.get('fuel_type_petrol'))
    seller_type_individual = bool(request.form.get('seller_type_individual'))
    transmission_manual = bool(request.form.get('transmission_manual'))

    # Perform the prediction
    features = [
        present_price,
        kms_driven,
        past_owners,
        age,
        fuel_type_diesel,
        fuel_type_petrol,
        seller_type_individual,
        transmission_manual
    ]
    prediction = model.predict([features])[0]

    result = f"The predicted selling price is: {prediction:.2f} lacs"

    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
