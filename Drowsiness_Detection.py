import time
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np
import pygame

# Initialize the mixer for sound alerts
mixer.init()
mixer.music.load("music.wav")

# Initialize pygame mixer for face direction alert
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("music.wav")  # Make sure "music.wav" exists in the same folder

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR) for yawning detection
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # Vertical
    B = distance.euclidean(mouth[4], mouth[8])   # Vertical
    C = distance.euclidean(mouth[0], mouth[6])   # Horizontal
    return (A + B) / (2.0 * C)

# Thresholds
EYE_THRESH = 0.25  # Adjusted threshold for drowsiness detection (EAR threshold)
MOUTH_THRESH = 0.6  # Yawning detection (MAR threshold)
DIRECTION_THRESH = 25  # Looking away detection (direction threshold)
ALERT_DELAY_EYE = 3  # 3-second delay for eye closure alert
ALERT_DELAY_YAWN = 2  # 3-second delay for yawning detection
ALERT_DELAY_FACE = 1 # 3-second delay for face detection alert
ALERT_DELAY_LOOKAWAY = 3  # 3-second delay for looking away alert

# Initialize timers
start_time_eye = None
start_time_yawn = None
start_time_face = None
start_time_lookaway = None

# Load the face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define landmarks for eyes, mouth, and jaw
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]

# Start video capture
cap = cv2.VideoCapture(0)

looking_straight_displayed = False  # Flag to track if "Looking Straight" is already displayed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    if subjects:
        if start_time_face is None:
            start_time_face = time.time()  # Start face detection timer

        elapsed_time_face = time.time() - start_time_face
        if elapsed_time_face >= ALERT_DELAY_FACE:
            cv2.putText(frame, "FACE DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract facial landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        jaw = shape[jStart:jEnd]

        # Compute EAR (Eye Aspect Ratio)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Compute MAR (Mouth Aspect Ratio)
        mar = mouth_aspect_ratio(mouth)

        # Calculate nose and jaw center for face direction detection
        nose = np.mean(shape[27:36], axis=0).astype(int)
        jaw_center = np.mean(shape[0:17], axis=0).astype(int)
        direction = nose[0] - jaw_center[0]

        # Draw facial contours (face boundary, eyes, mouth)
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 255), 1)
        cv2.drawContours(frame, [cv2.convexHull(jaw)], -1, (0, 0, 255), 1)

        # Draw the face boundary (rectangle around the face)
        (x, y, w, h) = (subject.left(), subject.top(), subject.width(), subject.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Eye Closure Detection (Drowsiness)
        if ear < EYE_THRESH:
            if start_time_eye is None:
                start_time_eye = time.time()  # Start timer when eyes first closed

            elapsed_time_eye = time.time() - start_time_eye
            if elapsed_time_eye >= ALERT_DELAY_EYE:
                print("DROWSINESS ALERT!")  # Alert for drowsiness
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
               
        else:
            start_time_eye = None  # Reset timer if eyes open

        # Yawning Detection
        if mar > MOUTH_THRESH:
            if start_time_yawn is None:
                start_time_yawn = time.time()  # Start timer when yawning detected

            elapsed_time_yawn = time.time() - start_time_yawn
            if elapsed_time_yawn >= ALERT_DELAY_YAWN:
                print("YAWNING DETECTED!")  # Alert for yawning
                cv2.putText(frame, "YAWNING DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                mixer.music.play()
               
        else:
            start_time_yawn = None  # Reset timer if no yawning detected

        # Face Direction Detection (Looking Away)
        if abs(direction) > DIRECTION_THRESH:
            if start_time_lookaway is None:
                start_time_lookaway = time.time()  # Start timer when looking away detected

            elapsed_time_lookaway = time.time() - start_time_lookaway
            if elapsed_time_lookaway >= ALERT_DELAY_LOOKAWAY:
                print("LOOKING AWAY!")  # Alert for looking away
                cv2.putText(frame, "LOOKING AWAY!",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_sound.play()
                looking_straight_displayed = False  # Reset flag so "Looking Straight" can be printed later
        else:
            start_time_lookaway = None  # Reset timer if not looking away
            if not looking_straight_displayed:
                print("Looking Straight")
                looking_straight_displayed = True  # Prevents multiple prints of "Looking Straight"

    # Display the frame
    cv2.imshow("Frame", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

