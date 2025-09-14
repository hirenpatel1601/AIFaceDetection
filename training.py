import numpy as np
import cv2
import os
import sqlite3
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataSet"
def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  # Convert to grayscale
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
        cv2.imshow("Training", imageNp)
        cv2.waitKey(10)

    return np.array(Ids), faces


ids, faces = getImagesWithID(path)
recognizer.train(faces, ids)
recognizer.save("recognizer/trainingData.yml")
cv2.destroyAllWindows()


facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")