
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fields = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
              'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']
    try:
        values = [float(request.form.get(f, 0)) for f in fields]
        prediction = model.predict([values])[0]
        result = "Chronic Kidney Disease Detected" if prediction == 1 else "No CKD Detected"
        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error occurred: {e}"
if __name__ == '__main__':
    app.run(debug=True)
