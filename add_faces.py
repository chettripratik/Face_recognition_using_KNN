# import cv2
# import pickle
# import numpy as np
# import os

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")


# faces_data = []
# i = 0

# name = input("Enter Your Name: ")


# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for x, y, w, h in faces:
#         crop_img =  frame[y: y+h, x: x+w, :]
#         resized_img = cv2.resize(crop_img, (50,50))
#         if len(faces_data)<=100 and i%10==0:
#             faces_data.append(resized_img)
#         i = i+1
#         cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255),1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)

#     if k == ord("q") or len(faces_data) == 100:
#         break

# video.release()
# cv2.destroyAllWindows()

# faces_data = np.asarray(faces_data)
# faces_data=faces_data.reshape(100, -1)



# if 'names.pkl' not in os.listdir('data/'):
#     names=[name]*100
#     with open('data/names.pkl' , 'wb') as f:
#         pickle.dump(names, f)
# else:
#     with open('data/names.pkl', 'rb') as f:
#         names=pickle.dump(f)

#         names = names+[name]*100
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)


# if 'faces_data' not in os.listdir('data/'):
    
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces_data, f)
# else:
#     with open('data/faces_data.pkl', 'rb') as f:
#         faces=pickle.dump(f)

#         faces = np.append(faces, faces_data, axis=0)
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces, f)

import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

faces_data = []
i = 0

name = input("Enter Your Name: ")

# --- 1. LOAD EXISTING DATABASE FOR DUPLICATE CHECK ---
check_knn = None  # Default to None

if os.path.exists('data/names.pkl') and os.path.exists('data/faces_data.pkl'):
    try:
        with open('data/names.pkl', 'rb') as f:
            LABELS = pickle.load(f)
        with open('data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)
        
        # Check if lists are not empty to avoid errors
        if len(LABELS) > 0 and len(FACES) > 0:
            check_knn = KNeighborsClassifier(n_neighbors=5)
            check_knn.fit(FACES, LABELS)
            print("Database loaded for duplicate check...")
        else:
            check_knn = None # Database empty
            
    except Exception as e:
        print(f"Database error: {e}")
        check_knn = None  # CRITICAL FIX: Ensure it is None if error occurs

speak(f"Please look at the camera to verify if you are already registered.")

# Load Background
try:
    imgBackground = cv2.imread('background.jpeg')
    if imgBackground is None: raise FileNotFoundError
except:
    imgBackground = np.zeros((480, 640, 3), dtype=np.uint8)

verified_new_user = False 

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        # --- 2. FACE DUPLICATION CHECK ---
        # We check only if we haven't started recording yet
        if not verified_new_user and check_knn is not None:
            try:
                prediction = check_knn.predict(resized_img)
                probability = check_knn.predict_proba(resized_img)
                confidence = np.max(probability)
                
                # If the model is very sure (>80%) that this face exists
                if confidence > 0.8:
                    existing_name = str(prediction[0])
                    print(f"Duplicate detected: {existing_name} ({confidence*100:.1f}%)")
                    speak(f"Face already registered as {existing_name}. Registration denied.")
                    
                    video.release()
                    cv2.destroyAllWindows()
                    exit()
                else:
                    verified_new_user = True
                    speak("Face not found in database. Starting capture.")
            except Exception as e:
                # If prediction fails, assume new user
                verified_new_user = True

        # If database is empty (check_knn is None), we skip the check
        if check_knn is None:
            verified_new_user = True

        # --- 3. CAPTURE LOGIC (Only runs if verified) ---
        if verified_new_user:
            resized_img_capture = cv2.resize(crop_img, (50, 50)) 
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img_capture)
            i += 1
            
            # Visuals
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            progress = len(faces_data)
            cv2.putText(frame, f"{progress}%", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Progress Bar
            cv2.rectangle(frame, (200, 400), (440, 420), (255, 255, 255), 1)
            bar_width = int(2.4 * progress)
            cv2.rectangle(frame, (200, 400), (200 + bar_width, 420), (0, 255, 0), -1)

            # Voice Feedback
            if progress == 50 and i % 10 == 0:
                speak("Halfway there.")
            if progress == 100 and i % 10 == 0:
                speak("Capture complete.")

    # Display Frame
    roi_x, roi_y, roi_w, roi_h = 65, 55, 480, 350
    frame_resized = cv2.resize(frame, (roi_w, roi_h))
    
    if imgBackground.shape[0] > roi_y + roi_h and imgBackground.shape[1] > roi_x + roi_w:
        imgBackground[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = frame_resized
        display_img = imgBackground
    else:
        display_img = frame

    cv2.imshow("Frame", display_img)
    
    k = cv2.waitKey(1)
    if k == ord("q") or len(faces_data) == 100:
        break

# ... (Upper part of code remains the same) ...

video.release()
cv2.destroyAllWindows()

# --- NEW SAFER SAVING LOGIC ---
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Ensure 'data' folder exists
os.makedirs('data', exist_ok=True)

# 1. Load Existing Data
if 'names.pkl' in os.listdir('data/') and 'faces_data.pkl' in os.listdir('data/'):
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
else:
    names = []
    faces = np.empty((0, faces_data.shape[1]))

# 2. Append New Data
names = names + [name] * 100
faces = np.append(faces, faces_data, axis=0)

# 3. SAFETY CHECK: Only save if lengths match
if len(names) == len(faces):
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
    print("Data Saved Successfully!")
    speak("Data saved successfully.")
else:
    print(f"Error: Data mismatch! Names: {len(names)}, Faces: {len(faces)}")
    print("Changes were NOT saved to prevent corruption.")
    speak("Error saving data.")