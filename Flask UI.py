# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the pre-trained model
# with open('C:\\Users\\kelvi\\Documents\\Big data analaytics\\SEM 3\\Neural networks and deep learning (AML 3104)\\Final Project\\Final Project\\tuned_decision_tree_model.pkl','rb') as model_file:
#     best_model = pickle.load(model_file)

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         stories = float(request.form['stories'])
#         fin_sqft = float(request.form['fin_sqft'])
#         bedrms = float(request.form['bedrms'])
#         baths = float(request.form['baths'])
#         lot_size = float(request.form['lot_size'])
#         Ald = float(request.form['Ald'])
#         age = float(request.form['age'])
#         full_base = float(request.form['full_base'])
#         attic = float(request.form['attic'])
#         fireplace = float(request.form['fireplace'])
#         air_conditioning = float(request.form['air_conditioning'])
#         garage = float(request.form['garage'])
#         x_coordinate = float(request.form['x_coordinate'])
#         y_coordinate = float(request.form['y_coordinate'])

#         # Make prediction using the model
#         features = np.array([[stories, fin_sqft, bedrms, baths, lot_size,Ald, age, full_base, attic,
#                               fireplace, air_conditioning, garage, x_coordinate, y_coordinate]])
#         prediction = best_model.predict(features)[0]

#         # Round the prediction to 2 decimal places
#         prediction = round(prediction, 2)

#         return render_template('result.html', prediction=prediction)
#     except ValueError as e:
#         return render_template('error.html', error_message=str(e))

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('C:\\Users\\kelvi\\Documents\\Big data analaytics\\SEM 3\\Neural networks and deep learning (AML 3104)\\Final Project\\Final Project\\tuned_decision_tree_model.pkl','rb') as model_file:
    best_model = pickle.load(model_file)

# Load the scaler
with open('C:\\Users\\kelvi\\Documents\\Big data analaytics\\SEM 3\\Neural networks and deep learning (AML 3104)\\Final Project\\Final Project\\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        stories = float(request.form['stories'])
        fin_sqft = float(request.form['fin_sqft'])
        bedrms = float(request.form['bedrms'])
        baths = float(request.form['baths'])
        lot_size = float(request.form['lot_size'])
        Ald = float(request.form['Ald'])
        age = float(request.form['age'])
        full_base = float(request.form['full_base'])
        attic = float(request.form['attic'])
        fireplace = float(request.form['fireplace'])
        air_conditioning = float(request.form['air_conditioning'])
        garage = float(request.form['garage'])
        x_coordinate = float(request.form['x_coordinate'])
        y_coordinate = float(request.form['y_coordinate'])

        # Scale the input features using the loaded scaler
        features = np.array([[stories, fin_sqft, bedrms, baths, lot_size, Ald, age, full_base, attic,
                              fireplace, air_conditioning, garage, x_coordinate, y_coordinate]])
        scaled_features = scaler.transform(features)

        # Make prediction using the model
        prediction = best_model.predict(scaled_features)[0]

        # Round the prediction to 2 decimal places
        prediction = round(prediction, 2)

        return render_template('result.html', prediction=prediction)
    except ValueError as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)





