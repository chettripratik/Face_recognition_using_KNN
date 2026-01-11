




# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from win32com.client import Dispatch

# def speak(str1):
#     speak = Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# # Load Labels and Faces
# with open('data/names.pkl', 'rb') as f:
#     LABELS = pickle.load(f)

# with open('data/faces_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# imgBackground = cv2.imread('background.jpeg')
# bg_h, bg_w, _ = imgBackground.shape

# COL_NAMES = ['NAME', 'TIME']

# # Dictionary to keep track of the last time attendance was taken for each person
# last_attendance_time = {}

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for x, y, w, h in faces:
#         crop_img = frame[y:y + h, x:x + w, :]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
#         # Predict the name
#         output = knn.predict(resized_img)
#         name = str(output[0])
        
#         # Calculate Confidence (Probability)
#         probabilities = knn.predict_proba(resized_img)
#         confidence = np.max(probabilities) * 100

#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

#         # Draw Rectangle and Name with Confidence
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
#         cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        
#         # Display Name + Confidence %
#         display_text = f"{name} {int(confidence)}%"
#         cv2.putText(frame, display_text, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

#         attendance = [name, str(timestamp)]

#         # --- AUTO ATTENDANCE LOGIC ---
#         # Check if this person has been marked in the last 60 seconds
#         if name not in last_attendance_time or (ts - last_attendance_time[name] > 60):
            
#             # Update the last attendance time for this user
#             last_attendance_time[name] = ts
            
#             # Personalized Voice Greeting
#             speak(f"Welcome {name}")
            
#             # Save Attendance to CSV
#             if exist:
#                 with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(attendance)
#             else:
#                 with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(COL_NAMES)
#                     writer.writerow(attendance)
            
#             print(f"Attendance marked for {name} at {timestamp}")

#     # Background Image Logic
#     roi_x = 65
#     roi_y = 55
#     roi_w = 480
#     roi_h = 350

#     frame_resized = cv2.resize(frame, (roi_w, roi_h))
#     imgBackground[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = frame_resized

#     cv2.imshow("Frame", imgBackground)

#     k = cv2.waitKey(1)

#     if k == ord("q"):
#         break

# video.release()
# cv2.destroyAllWindows()



from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread('background.jpeg')

COL_NAMES = ['NAME', 'TIME']

# --- Load existing attendance for today to prevent duplicates ---
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
file_path = "Attendance/Attendance_" + date + ".csv"
attendees_today = set()

if os.path.isfile(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if row:
                attendees_today.add(row[0])

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Variables to manage the "Auto Close" logic
    name_to_speak = None
    should_close = False

    for x, y, w, h in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        name = str(output[0])
        
        # Calculate Confidence
        probabilities = knn.predict_proba(resized_img)
        confidence = np.max(probabilities) * 100

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        # --- LOGIC ---
        if name in attendees_today:
            # ALREADY MARKED
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
            cv2.putText(frame, f"{name} (Marked)", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            
            # Prepare to speak and close
            name_to_speak = f"{name} is already marked"
            should_close = True
        else:
            # NEW ATTENDANCE
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, f"{name} {int(confidence)}%", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Save to CSV immediately
            attendance = [name, str(timestamp)]
            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
            
            attendees_today.add(name)
            
            # Prepare to speak and close
            name_to_speak = f"Welcome {name}"
            should_close = True

    # --- UI UPDATE ---
    roi_x = 65
    roi_y = 55
    roi_w = 480
    roi_h = 350

    frame_resized = cv2.resize(frame, (roi_w, roi_h))
    imgBackground[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = frame_resized

    cv2.imshow("Frame", imgBackground)
    cv2.waitKey(1) # Updates the screen to show the box

    # --- ACTION PHASE ---
    if name_to_speak:
        speak(name_to_speak) # This blocks, so we do it AFTER showing the box
    
    if should_close:
        time.sleep(1) # Brief pause so you can see the "Marked" status
        break

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()