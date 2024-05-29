import numpy as np 
import face_recognition
import cv2
import os
import RPi.GPIO as GPIO
import time

# Setup GPIO
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 23
STEPPER_PINS = [17, 27, 22, 5]  # Example GPIO pins for the stepper motor
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
for pin in STEPPER_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Define full step sequence
fullstep_seq = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

# Function to move stepper motor
def move_stepper(steps, delay=0.01):
    for _ in range(steps):
        for fullstep in range(4):
            for pin in range(4):
                GPIO.output(STEPPER_PINS[pin], fullstep_seq[fullstep][pin])
            time.sleep(delay)

path = 'person'
images = []
classNames = []
personlist = os.listdir(path)

print(personlist)

for cl in personlist:
    if cl.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
        curperson = cv2.imread(os.path.join(path, cl))
        images.append(curperson)
        classNames.append(os.path.splitext(cl)[0])

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img, None, 1, "small")[0]
        encodeList.append(encode)
    return encodeList 

encodeListKnown = find_encodings(images)
print(encodeListKnown)

cap = cv2.VideoCapture(0)
door_closed = True
door_open_time = None

while True:
    _, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)
    
    for encode_face, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encode_face)
        faceDis = face_recognition.face_distance(encodeListKnown, encode_face)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            if door_closed:
                move_stepper(900)  # Move stepper motor to open the door
                door_closed = False
                door_open_time = time.time()
        else:
            name = "Unknown"
            
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Check for button press to manually open the door
    button_state = GPIO.input(BUTTON_PIN)
    if button_state == GPIO.LOW and door_closed:
        move_stepper(900)  # Move stepper motor to open the door
        door_closed = False
        door_open_time = time.time()

    # Check if the door should be closed after 5 seconds
    if door_open_time and time.time() - door_open_time > 5:
        move_stepper(900)  # Move stepper motor to close the door
        door_closed = True
        door_open_time = None
    
    cv2.imshow('Face Recognition', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Clean up GPIO
GPIO.cleanup()
