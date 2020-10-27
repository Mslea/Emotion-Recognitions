from flask import Flask, request
import json
import base64
from PIL import Image
import io
import numpy as np
from emotion import get_emotion
import time
app = Flask(__name__)

@app.route('/')
def welcome():
    return "welcome"
@app.route('/emotion_recognitions',methods = ["POST"])
def recognitions():
    if request.method == 'POST':
        data = request.get_json()
        x = data["base64Image"].split(',')
        base64_decoded = base64.b64decode(x[1])
        image = Image.open(io.BytesIO(base64_decoded))
        image_np = np.array(image)
        t = time.time()
        result = get_emotion(image_np)
        print(result)
        print(time.time() - t)
        return json.dumps(result)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)