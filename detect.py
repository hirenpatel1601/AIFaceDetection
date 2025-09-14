import cv2
import numpy as np
import os
import sqlite3
from PIL import Image
import pyttsx3


engine = pyttsx3.init()
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")   
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Could not access the camera")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingData.yml")  

def getProfile(Id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM students WHERE Id=?", (Id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

welcomed = set()  # ✅ store names already spoken

while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame from camera")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    names_in_frame = []

    for (x, y, w, h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 50:
            profile = getProfile(Id)
            if profile is not None:
                cv2.putText(img, f"Name: {profile[1]}", (x, y-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, f"Age: {profile[2]}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Add name to list
                names_in_frame.append(profile[1])
        else:
            cv2.putText(img, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # ✅ Speak after all faces are processed
    if names_in_frame:
        unique_names = set(names_in_frame) - welcomed  # only new names
        if unique_names:
            text = "Welcome " + ", ".join(unique_names)
            engine.say(text)
            engine.runAndWait()
            welcomed.update(unique_names)  # mark as spoken

    cv2.imshow("Face", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
engine.stop()