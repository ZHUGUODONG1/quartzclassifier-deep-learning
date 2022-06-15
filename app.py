from tensorflow.keras.models import load_model
import numpy as np
from flask import request ,Flask,render_template
from sklearn import preprocessing
app=Flask(__name__)
path='shiying.h5'
model=load_model(path)


@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST","GET"])
def predict():
    X = np.loadtxt('Quartzdataset.csv', delimiter=",", usecols=(1, 2, 3, 4, 5), dtype=float, skiprows=1)
    dtype = ['IRG', 'carlin', 'epithermal', 'granite', 'greisen', 'orogenic', 'pegmatite', 'porphyry', 'skarn']
    a=[]
    float_features=[float(x)for x in request.form.values()]
    features=np.array(float_features,dtype=float)
    X = np.row_stack((X, features))
    X = np.log(X + 1)
    X = preprocessing.scale(X)  # x是要进行标准化的样本数据
    features=X[-1,:]
    a.append(features)
    a = np.array(a, float)
    prediction=model.predict(a)
    prediction = np.argmax(prediction, axis=1)
    # return dtype[prediction[0]]
    prediction=dtype[prediction[0]]
    # print(prediction)
    return render_template("index.html",prediction_text=prediction)

if __name__ =="__main__":
    app.run(debug=True)