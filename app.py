import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
import sklearn
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model32 = pickle.load(open('parkinson_model32.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')

scaler = MinMaxScaler()
@app.route('/predict', methods=['POST'])
def predict_parkinson():
    feature1 = str(request.form.get('MDVP:Fo(Hz)'))
    feature2 = str(request.form.get('MDVP:Fhi(Hz)'))
    feature3 = str(request.form.get('MDVP:Flo(Hz)'))
    feature4 = str(request.form.get('MDVP:Jitter(%)'))
    feature5 = str(request.form.get('MDVP:Jitter(Abs)'))
    feature6 = str(request.form.get('MDVP:RAP'))
    feature7 = str(request.form.get('MDVP:PPQ'))
    feature8 = str(request.form.get('Jitter:DDP'))
    feature9 = str(request.form.get('MDVP:Shimmer'))
    feature10 = str(request.form.get('MDVP:Shimmer(dB)'))
    feature11 = str(request.form.get('Shimmer:APQ3'))
    feature12 = str(request.form.get('Shimmer:APQ5'))
    feature13 = str(request.form.get('MDVP:APQ'))
    feature14 = str(request.form.get('Shimmer:DDA'))
    feature15 = str(request.form.get('NHR'))
    feature16 = str(request.form.get('HNR'))
    feature17 = str(request.form.get('RPDE'))
    feature18 = str(request.form.get('DFA'))
    feature19 = str(request.form.get('spread1'))
    feature20 = str(request.form.get('spread2'))
    feature21 = str(request.form.get('D2'))
    feature22 = str(request.form.get('PPE'))

    values = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22]])

    result = model32.predict(values)

    if result[0]==1:
        result = 'Positive Report' 
    else:
        result = 'Negative Report'
        
    return render_template('index.html', result=result)

if __name__=='__main__':
    app.run(debug=True)
