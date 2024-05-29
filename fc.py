import os
import RPi.GPIO as GPIO
import time
import cv2
import face_recognition

# Set up GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins
motor_pins = [17, 18, 27, 22]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Push button pin
button_pin = 23
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Function to run the motor for 900 steps
def move_motor(steps=900):
    seq = [[1,0,0,1], [1,0,0,0], [1,1,0,0], [0,1,0,0], [0,1,1,0], [0,0,1,0], [0,0,1,1], [0,0,0,1]]
    for _ in range(abs(steps)):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(motor_pins[pin], seq[halfstep][pin])
            time.sleep(0.001)
    if steps < 0:  # If steps is negative, reverse the sequence
        seq.reverse()

# Open and close door
def open_door():
    move_motor(900)  # Move 900 steps to open the door
    time.sleep(5)  # Keep the door open for 5 seconds
    move_motor(-900)  # Move -900 steps to close the door

# Load known faces from the "person" directory
known_face_encodings = []
known_face_names = []

person_directory = "person"
for filename in os.listdir(person_directory):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Adjust file extensions as needed
        filepath = os.path.join(person_directory, filename)
        image = face_recognition.load_image_file(filepath)
        face_encoding = face_recognition.face_encodings(image)
        if face_encoding:
            known_face_encodings.append(face_encoding[0])
            name, _ = os.path.splitext(filename)
            known_face_names.append(name)

# Camera setup
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        if name != "Unknown":
            open_door()

    # Check if push button is pressed
    if GPIO.input(button_pin) == GPIO.LOW:
        open_door()

    # Display the results (optional)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
GPIO.cleanup()
