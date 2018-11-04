from flask import Flask, render_template, request
# library for reading,saving,resizing images
from scipy.misc import imread, imresize
# matrix math
import numpy as np
# regular expressions
import re
# system functions
import sys
# os functions
import os

# where our model data is
sys.path.append(os.path.abspath("./model"))

from load import *
import base64

app = Flask(__name__)
# global vars model and graph
global model, graph
model, graph = init()

# decode an image from base64 into raw representation
def convertImage(imgData):
    imgstr = re.search(r'base64,(.*)',str(imgData)).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    print('predict called')
    # get img data from request object
    imgData1 = request.get_data()
    # convert to png
    convertImage(imgData1)
    # read image file
    x = imread('output.png', mode='L')
    x = imresize(x, (28,28))
    # save final image
    # imsave('final_image.jpg', x)
    # convert to 4D tensor for model
    x = x.reshape(1, 28, 28, 1)
    # computation graph
    with graph.as_default():
        # predict
        out = model.predict(x)
        print(out)
        # convert response to string
        response = np.argmax(out, axis=1)
        print(response)
        return str(response[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
