from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np

# ML PACKAGES
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)
    loaded_model = pickle.load(open("data/model_1","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]



@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        if int(result)==1:
            prediction = result.round()   # 'Income more than 50K'
        else:
            prediction = result.round(2)       # 'Income less than 50K'
        # if prediction ==1 :
        #
        # else:
        #     prediction == 0

        return render_template("results.html",prediction=prediction)



    # df = pd.read_csv("data/HousePrices.csv")
    # df_x = df.Area.astype(int)
    # df_y = df.Price
    #
    # corpus = df_x
    # cv = CountVectorizer()
    # X = cv.fit_transform(corpus)
    #
    # p_model = open('data/model_2', 'rb')
    # clf = joblib.load(p_model)

    # if request.method == 'POST':
    #     comment = request.form['comment']
    #     data = [comment]
    #     vect = cv.transform(data).toarray()
    #     my_prediction = clf.predict(vect)
    # return render_template('results.html',prediction = my_prediction, Area = int(comment))






if __name__ == '__main__':
    app.run(debug=True)
