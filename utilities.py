import numpy as np
import pandas as pd
import mediapipe as mp
import cv2

mediapipe_pose = mp.solutions.pose

# Calculates the angle at p2 by p1 and p3 joints
def calculate_joint_angle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2) 
    p3 = np.array(p3)

    angle_in_radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle_in_degrees = np.abs(angle_in_radians * 180.0 / np.pi)

    if angle_in_degrees > 180.0:
        angle_in_degrees = 360 - angle_in_degrees

    return angle_in_degrees


# Detect body part joint given the landmark and joint name
def detect_joint(lm, joint_name):
    return [
        lm[mediapipe_pose.PoseLandmark[joint_name].value].x,
        lm[mediapipe_pose.PoseLandmark[joint_name].value].y,
        # lm[mediapipe_pose.PoseLandmark[joint_name].value].visibility
    ]


# Detect body parts on the basis of landmark
def detect_joints(lm):
    body_parts = pd.DataFrame(columns=["x", "y"])

    for i, lk in enumerate(mediapipe_pose.PoseLandmark):
        lk = str(lk).split(".")[1]
        cord = detect_joint(lm, lk)
        body_parts.loc[lk] = cord[0], cord[1]

    return body_parts


# Utility function for displaying the attributes on screen
def display_table(frame , counter, stage, msg):
    lines = str(msg).split("\n")
    x, y = 15, 140

    for line in lines:
        cv2.rectangle(frame, (0, 0), (600, y + 10), (245, 117, 16), -1)
        y += 30

    y = 140

    for line in lines:
        cv2.putText(frame, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2, cv2.LINE_AA,
        )
        y += 30

    cv2.putText(frame, "REPEAT", (15, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(frame, str(counter), (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(frame, "STAGE", (125, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 0, 0), 1,cv2.LINE_AA,
    )
    cv2.putText(frame, str(stage), (125, 60), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(frame, "CORRECTION", (15, 100), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 0, 0), 1, cv2.LINE_AA,
    )
    return frame
    
