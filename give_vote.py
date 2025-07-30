from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

min_samples = min(len(FACES), len(LABELS))
FACES, LABELS = FACES[:min_samples], LABELS[:min_samples]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

script_dir = os.path.dirname(os.path.abspath(__file__))  
background_path = os.path.join(script_dir, "resources", "background.png")  

imgBackground = cv2.imread(background_path)

if imgBackground is None:
    print("Error: 'background.png' not found or could not be loaded.")
    exit()

def find_white_space(img):
    mask = cv2.inRange(img, (240, 240, 240), (255, 255, 255))  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h  
    return None

white_space = find_white_space(imgBackground)
if white_space:
    x, y, w, h = white_space
    scale_factor = 0.7
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    x_offset = x + (w - new_w) // 2
    y_offset = y + (h - new_h) // 2
else:
    x, y, w, h = 255, 370, 640, 173

COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Votes.csv")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    resized_frame = cv2.resize(frame, (new_w, new_h))
    imgBackground[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    cv2.imshow('frame', imgBackground)
    k = cv2.waitKey(1)

    def check_if_exists(value):
        try:
            with open("Votes.csv", 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0] == value:
                        return True
        except FileNotFoundError:
            print("File not found or unable to open the CSV file.")
        return False

    voter_exist = check_if_exists(output[0])
    if voter_exist:
        print("YOU HAVE ALREADY VOTED")
        speak("YOU HAVE ALREADY VOTED")
        break

    if k == ord('1'):
        speak("YOUR VOTE HAS BEEN RECORDED")
        time.sleep(3)
        if exist:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                attendance = [output[0],'BJP',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                attendance = [output[0],'BJP',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
        break

    if k == ord('2'):
        speak("YOUR VOTE HAS BEEN RECORDED")
        time.sleep(3)
        if exist:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                attendance = [output[0],'CONGRESS',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                attendance = [output[0],'CONGRESS',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
        break

    if k == ord('3'):
        speak("YOUR VOTE HAS BEEN RECORDED")
        time.sleep(3)
        if exist:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                attendance = [output[0],'AAP',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                attendance = [output[0],'AAP',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
        break

    if k == ord('4'):
        speak("YOUR VOTE HAS BEEN RECORDED")
        time.sleep(3)
        if exist:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                attendance = [output[0],'NOTA',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Votes" + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                attendance = [output[0],'NOTA',date,timestamp]
                writer.writerow(attendance)
            csvfile.close()
        speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
        break

video.release()
cv2.destroyAllWindows()