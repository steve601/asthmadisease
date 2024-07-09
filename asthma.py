from flask import Flask,request,render_template
import pandas as pd
import pickle

app = Flask(__name__)
def load_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model
model_path = 'elements\model.pkl'
scaler_path = 'elements\scaler.pkl'

model = load_object(model_path)
scaler = load_object(scaler_path)

@app.route('/')
def homepage():
    return render_template('asthma.html')

@app.route('/classify',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    
    columns = ['smoking', 'sleepquality', 'pollenexposure', 'dustexposure',
       'petallergy', 'hayfever', 'gastroesophagealreflux', 'lungfunctionfev1',
       'lungfunctionfvc', 'wheezing', 'shortnessofbreath', 'chesttightness',
       'coughing', 'nighttimesymptoms', 'exerciseinduced']
    features = pd.DataFrame([features],columns=columns)
    features = scaler.transform(features)
    result = model.predict(features)
    
    msg = 'The patient has asthma' if result == 1 else 'Patient has no asthma'
    
    return render_template('asthma.html',text = msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0")