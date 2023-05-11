import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
import sklearn
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model32 = pickle.load(open('parkinson_model32.pkl','rb'))
@app.route('/')
def Home():
    return render_template('index.html')


scaler = MinMaxScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        feature1 = float(request.form['MDVP:Fo(Hz)'])
        feature2 = float(request.form['MDVP:Fhi(Hz)'])
        feature3 = float(request.form['MDVP:Flo(Hz)'])
        feature4 = float(request.form['MDVP:Jitter(%)'])
        feature5 = float(request.form['MDVP:Jitter(Abs)'])
        feature6 = float(request.form['MDVP:RAP'])
        feature7 = float(request.form['MDVP:PPQ'])
        feature8 = float(request.form['Jitter:DDP'])
        feature9 = float(request.form['MDVP:Shimmer'])
        feature10 = float(request.form['MDVP:Shimmer(dB)'])
        feature11 = float(request.form['Shimmer:APQ3'])
        feature12 = float(request.form['Shimmer:APQ5'])
        feature13 = float(request.form['MDVP:APQ'])
        feature14 = float(request.form['Shimmer:DDA'])
        feature15 = float(request.form['NHR'])
        feature16 = float(request.form['HNR'])
        feature17 = float(request.form['RPDE'])
        feature18 = float(request.form['DFA'])
        feature19 = float(request.form['spread1'])
        feature20 = float(request.form['spread2'])
        feature21 = float(request.form['D2'])
        feature22 = float(request.form['PPE'])

        values = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22]).reshape(1,3)

        prediction = model32.predict(values)

        if (prediction[0]==1):
            return render_template('index.html', prediction_text='positive report') 

        else:
            return render_template('index.html', prediction_text='negative report')

    else:
        return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)
