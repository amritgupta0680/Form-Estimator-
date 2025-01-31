import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def evaluate_position(angle, exercise_type):
    if exercise_type == "Squats":
        if 70 <= angle <= 160:
            return True, "Perfect squat form!"
        elif angle < 70:
            return False, "You need to go lower in your squat!"
        else:
            return False, "Extend fully at the top of your squat!"
    elif exercise_type == "Push-ups":
        if 80 <= angle <= 160:
            return True, "Great push-up form!"
        elif angle < 80:
            return False, "You need to lower your chest more during push-ups!"
        else:
            return False, "Fully extend your arms during push-ups!"
    elif exercise_type == "Pull-ups":
        if 90 <= angle <= 160:
            return True, "Perfect pull-up form!"
        elif angle < 90:
            return False, "Try to pull higher during the pull-up!"
        else:
            return False, "You should not pull past this angle. Maintain proper form."

def analyze_video(filepath, exercise_type):
    cap = cv2.VideoCapture(filepath)
    angles = []
    feedback = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]
            hip = [landmarks[23].x, landmarks[23].y]
            knee = [landmarks[25].x, landmarks[25].y]
            ankle = [landmarks[27].x, landmarks[27].y]
            
            if exercise_type == "Squats":
                angle = calculate_angle(hip, knee, ankle)
                angles.append(angle)
            elif exercise_type == "Push-ups":
                angle = calculate_angle(shoulder, elbow, wrist)
                angles.append(angle)
            elif exercise_type == "Pull-ups":
                angle = calculate_angle(shoulder, elbow, wrist)
                angles.append(angle)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cap.release()

    if angles:
        avg_angle = np.mean(angles)  # Calculate the average angle for the entire video
        is_perfect, correction = evaluate_position(avg_angle, exercise_type)
        feedback.append(f"Average {exercise_type} angle: {avg_angle:.2f}")
        feedback.append(correction)
    else:
        feedback.append("No feedback could be generated. Ensure proper visibility and upload again.")
    return feedback

st.set_page_config(page_title="Exercise Form Analyzer", page_icon="🏋️", layout="centered")

st.title("🏋️ Exercise Form Analyzer")
st.markdown("""
Analyze your exercise form with AI-powered feedback!  
Upload a video of yourself performing exercises like **Squats**, **Push-ups**, or **Pull-ups**, and receive actionable insights to improve your technique.
""")

exercise_type = st.selectbox("Select Exercise Type:", ["Squats", "Push-ups", "Pull-ups"], index=0)

uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
analyze_button = st.button("Analyze My Form")

if uploaded_file and analyze_button:
    temp_path = os.path.join("temp_video.mp4")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.video(temp_path)
    st.markdown("### Feedback:")
    feedback = analyze_video(temp_path, exercise_type)
    os.remove(temp_path)
    if feedback:
        for comment in feedback:
            st.markdown(f"- {comment}")
    else:
        st.markdown("No feedback could be generated. Ensure proper visibility and upload again.")
