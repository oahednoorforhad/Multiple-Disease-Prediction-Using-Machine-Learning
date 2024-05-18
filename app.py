import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/disease_prediction')
def disease_prediction():
    return render_template('disease_prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Uncomment if you need an API endpoint for predictions
# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
#     new_data = scaler.transform(np.array(data).reshape(1, -1))
#     output = knn_model.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = svm_model.predict(final_input)[0]
    
    # Customize the output message
    if output == 'Healthy':  # Assuming the model predicts 'healthy' for healthy cases
        prediction_text = "You are absolutely Healthy"
        return render_template("disease_prediction.html", prediction_text=prediction_text)
    elif output == 'Diabetes':
        prediction_text = "You are suffering From Diabetes"
        return render_template("diabetes.html", prediction_text=prediction_text)
    elif output == 'Thelassemia':
        prediction_text = "You are suffering From Anemia"
        return render_template("thelassemia.html", prediction_text=prediction_text)
    elif output == 'Anemia':
        prediction_text = "You are suffering From Anemia"
        return render_template("anemia.html", prediction_text=prediction_text)
    elif output == 'Thromboc':
        prediction_text = "You are suffering From Thrombocytopenia"
        return render_template("thromboc.html", prediction_text=prediction_text)
    else:
        prediction_text = "You are suffering from {}".format(output)
    
    return render_template("disease_prediction.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)