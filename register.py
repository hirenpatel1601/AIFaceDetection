# import streamlit as st
# import pandas as pd
# import numpy as np
# from PIL import Image
# import io
# import cv2
# import os
# import sqlite3

# st.set_page_config(page_title="Students Registration", page_icon=":guardsman:", layout="wide")

# # ===================== SIDEBAR =====================
# left_column, right_column = st.columns([1,3])


# with left_column:
#     st.markdown("## Menu")
#     page = st.radio(
#         label="Select page",
#         options=["Home", "Upload Image", "Upload CSV", "Visualization", "Settings"],
#     )
#     st.markdown("---")
#     st.markdown("Made with :blue[Streamlit]")




# with right_column:
#      # ===================== HOME PAGE =====================
#     if page == "Home":
#         st.title("🎓 Student Enrollment – Face Detection Attendance System")

#         st.markdown("Fill the form and capture student face for registration.")
#         student_id = st.text_input("Student ID")
#         student_name = st.text_input("Student Name")
#         students_age = st.number_input("Students Age",min_value=1,max_value=100,step=1)
#         facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#         if student_id and student_name and students_age:
#             st.success("Enter details and click *Capture Face* to enroll.")

#             # Webcam capture in Streamlit
#             img_file_buffer = st.camera_input("Capture Student Face")
#             #img_file_buffer = st.camera_input("Capture Student Face")
#             cam = cv2.VideoCapture(0)
#             if not cam.isOpened():
#                 print("❌ Error: Could not access the camera")
#                 exit()
#             # Create dataset folder if not exists
#             if not os.path.exists("dataSet"):
#                 os.makedirs("dataSet")

#             # Database function
#             def insert_or_update(Id, Name, Age):
#                 conn = sqlite3.connect("sqlite.db")
#                 cursor = conn.execute("SELECT * FROM students WHERE Id=?", (Id,))
#                 isRecordExist = cursor.fetchone()

#                 if isRecordExist:
#                     conn.execute("UPDATE students SET Name=?, Age=? WHERE Id=?", (Name, Age, Id))
#                 else:
#                     conn.execute("INSERT INTO students(Id, Name, Age) VALUES(?,?,?)", (Id, Name, Age))

#                 conn.commit()
#                 conn.close()
            
#             insert_or_update(student_id, student_name, students_age)
#             sampleNum = 0
#             print("📸 Capturing face samples. Look at the camera...")

#             while True:
#                 ret, img = cam.read()
#                 if not ret:
#                     print("❌ Failed to grab frame from camera")
#                     break

#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 faces = facedetect.detectMultiScale(gray, 1.3, 5)

#                 for (x, y, w, h) in faces:
#                     sampleNum += 1
#                     cv2.imwrite(f"dataSet/User.{student_id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
#                     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                     cv2.waitKey(100)

#                 cv2.imshow("Face", img)

#                 # Press 'q' to quit early
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                  break

#                 if sampleNum >= 20:
#                     print("✅ Face samples collected successfully")
#                     break

#                 cam.release()
#                 cv2.destroyAllWindows()




import streamlit as st
import cv2
import os
import numpy as np
import sqlite3
from PIL import Image
import pyttsx3

st.set_page_config(page_title="AI Attendance Enrollment", layout="wide")

# Ensure dataset & trainer folders
os.makedirs("dataSet", exist_ok=True)
os.makedirs("trainer", exist_ok=True)

# Database function
def insert_or_update(Id, Name, Age):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM students WHERE Id=?", (Id,))
    isRecordExist = cursor.fetchone()

    if isRecordExist:
        conn.execute("UPDATE students SET Name=?, Age=? WHERE Id=?", (Name, Age, Id))
    else:
        conn.execute("INSERT INTO students(Id, Name, Age) VALUES(?,?,?)", (Id, Name, Age))
    conn.commit()
    conn.close()

# Face detector
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("🎓 Student Enrollment – Face Detection Attendance")

# ----------------- TRAIN BUTTON -----------------
if st.button("🚀 Train Model"):
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

if st.button("🚀 Detect"):
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


    ids, faces = getImagesWithID(path)
    recognizer.train(faces, ids)
    recognizer.save("recognizer/trainingData.yml")
    cv2.destroyAllWindows()
# ----------------- ENROLLMENT FORM -----------------
student_id = st.text_input("Student ID")
student_name = st.text_input("Student Name")
students_age = st.number_input("Age", min_value=1, max_value=100)

if student_id and student_name and students_age:
    insert_or_update(student_id, student_name, students_age)

    if f"user_{student_id}_count" not in st.session_state:
        st.session_state[f"user_{student_id}_count"] = 0

    count = st.session_state[f"user_{student_id}_count"]

    if count < 20:
        st.warning(f"Capture {20-count} more photos for {student_name}.")
        img_file_buffer = st.camera_input("📸 Capture Face")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                st.error("❌ No face detected, try again.")
            else:
                for (x, y, w, h) in faces:
                    face_img = gray[y:y+h, x:x+w]
                    count += 1
                    filename = f"dataSet/User.{student_id}.{count}.jpg"
                    cv2.imwrite(filename, face_img)

                    st.session_state[f"user_{student_id}_count"] = count
                    st.image(cv2_img[y:y+h, x:x+w], caption=f"Face {count}/20", channels="GRAY")
                    st.success(f"✅ Saved {filename}")

    else:
        st.success(f"🎉 Completed! 20 face samples saved for {student_name} (ID: {student_id})")


