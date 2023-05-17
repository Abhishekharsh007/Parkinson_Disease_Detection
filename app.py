import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
import sklearn
import pickle
import decimal
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model32 = pickle.load(open('parkinson_model32.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_parkinson():
    feature1 = float(request.form.get('MDVP:Fo(Hz)'))
    feature2 = float(request.form.get('MDVP:Fhi(Hz)'))
    feature3 = float(request.form.get('MDVP:Flo(Hz)'))
    feature4 = float(request.form.get('MDVP:Jitter(%)'))
    feature5 = float(request.form.get('MDVP:Jitter(Abs)'))
    feature6 = float(request.form.get('MDVP:RAP'))
    feature7 = float(request.form.get('MDVP:PPQ'))
    feature8 = float(request.form.get('Jitter:DDP'))
    feature9 = float(request.form.get('MDVP:Shimmer'))
    feature10 = float(request.form.get('MDVP:Shimmer(dB)'))
    feature11 = float(request.form.get('Shimmer:APQ3'))
    feature12 = float(request.form.get('Shimmer:APQ5'))
    feature13 = float(request.form.get('MDVP:APQ'))
    feature14 = float(request.form.get('Shimmer:DDA'))
    feature15 = float(request.form.get('NHR'))
    feature16 = float(request.form.get('HNR'))
    feature17 = float(request.form.get('RPDE'))
    feature18 = float(request.form.get('DFA'))
    feature19 = float(request.form.get('spread1'))
    feature20 = float(request.form.get('spread2'))
    feature21 = float(request.form.get('D2'))
    feature22 = float(request.form.get('PPE'))

    values = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22]).reshape(1, -1)
    values = scaler.transform(values)
    
    preresult = model32.predict(values)
    result = round(preresult[0], 2)

    if (result==0):
        return render_template('index.html', result='Negative')
    else:
        return render_template('index.html', result='Positive')

if __name__=='__main__':
    app.run(debug=True)
