from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
from mtcnn import MTCNN
import json
detector = MTCNN()
# parameters for loading data and images
emotion_model_path = 'model/_mini_XCEPTION.102-0.66.hdf5'

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
#emotion
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

def predict_emotion(face):
    roi = cv2.resize(face,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi)[0]

    return dict((key, float(value)) for (key,value) in zip(EMOTIONS, preds))

def get_emotion(image):
    r = detector.detect_faces(image)
    for faces in r:
        left , top, width , height = faces['box']
        face = image[top: (top+ height), left: (left + width)].copy()
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        emo = predict_emotion(face)
        faces['emotion'] = emo
    return json.dumps(r)