import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model_logreg = pickle.load(open('model2.pkl', 'rb'))
_vector = pickle.load(open('vectorizer.pkl', 'rb'))
#cv = pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    to_predict_list = request.form['Text']
    #to_predict_list = np.asarray(to_predict_list.values())
    #print(to_predict_list)

    #vectorizer = TfidfVectorizer(max_features=23585)
    #vectorizer.fit([_vector])
    to_predict_list = _vector.transform([to_predict_list])

    pred = model_logreg.predict(to_predict_list)
    prediction_logreg = "Logistic Regression: "+str(pred[0])
    if(pred==1):
        pred_result ="The News is real"
    else:
        pred_result = "The News is fake"
    return render_template('index.html',prediction_text="Below are prediction results: ", prediction_text_logreg= pred_result)


if __name__ == "__main__":
    app.run(debug=True)