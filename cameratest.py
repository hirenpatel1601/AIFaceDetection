import cv2
import sqlite3
import os

# Load Haar Cascade (use absolute path for safety)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Could not access the camera")
    exit()

# Create dataset folder if not exists
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

# Get user details
Id = input("Enter your Id: ")
Name = input("Enter your Name: ")
Age = input("Enter your Age: ")
insert_or_update(Id, Name, Age)

sampleNum = 0
print("📸 Capturing face samples. Look at the camera...")

while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame from camera")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"dataSet/User.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)

    cv2.imshow("Face", img)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if sampleNum >= 20:
        print("✅ Face samples collected successfully")
        break

cam.release()
cv2.destroyAllWindows()
