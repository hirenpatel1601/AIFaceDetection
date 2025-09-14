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

st.set_page_config(page_title="AI Attendance Enrollment", layout="wide")

# Ensure dataset folder
if not os.path.exists("dataSet"):
    os.makedirs("dataSet")

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

        img_file_buffer = st.camera_input("Capture Face")

        if img_file_buffer is not None:
            # Convert image
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


