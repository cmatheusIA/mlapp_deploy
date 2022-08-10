
from flask import Flask,request
from utils import predict_pipe

app = Flask(__name__)

@app.route('/predict',methods = ['POST'])
def predict():
    data = request.data
    result = predict_pipe(data)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=False)