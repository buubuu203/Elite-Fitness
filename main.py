import cv2
import mediapipe as mp
import numpy as np, pandas as pd
from utilities import *
from joint_angles import JointAngle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture("Exercise_Videos/standard.mp4")
# cap = cv2.VideoCapture("Exercise_Videos/low-hands.MP4")
cap = cv2.VideoCapture("Exercise_Videos/misaligned-hands.MP4")
# cap = cv2.VideoCapture("Exercise_Videos/narrow-legs.MP4")
df = pd.read_csv('./dataset/velocities.csv').drop('Unnamed: 0', axis=1)
velocities = df.to_dict()
angles = {
    'leftArm': [],
    'rightArm': [],
}

# Curl counter variables
counter = 0
stage = None
msg = None
frameNo = 0
n = 2
angularVelocity = 0

Left_Leg_Angle = 0
Right_Leg_Angle = 0
Neck_Angle = 0
Left_Arm_Angle = 0
Right_Arm_Angle = 0
Abdomen_Angle = 0
Back_Angle = 0


# Setup Mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #  Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            left_hip = detect_joint(landmarks, 'LEFT_HIP')
            right_hip = detect_joint(landmarks, 'RIGHT_HIP')
            hip = [(right_hip[0] + left_hip[0]) / 2,
                   (right_hip[1] + left_hip[1]) / 2]
            left_wrist = detect_joint(landmarks, 'LEFT_WRIST')
            right_wrist = detect_joint(landmarks, 'RIGHT_WRIST')
            left_knee = detect_joint(landmarks, 'LEFT_KNEE')
            right_knee = detect_joint(landmarks, 'RIGHT_KNEE')
            knee = [
                (left_knee[0] + right_knee[0]) / 2,
                (left_knee[1] + right_knee[1]) / 2,
            ]
            left_shoulder = detect_joint(landmarks, 'LEFT_SHOULDER')
            right_shoulder = detect_joint(landmarks, 'RIGHT_SHOULDER')
            shoulder = [
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
            ]
            neck = shoulder
            ref = [(hip[0] + shoulder[0]) / 2, (hip[1] + shoulder[1]) / 2]
            left_elbow = detect_joint(landmarks, 'LEFT_ELBOW')
            right_elbow = detect_joint(landmarks, 'RIGHT_ELBOW')
            left_ankle = detect_joint(landmarks, 'LEFT_ANKLE')
            right_ankle = detect_joint(landmarks, 'RIGHT_ANKLE')
            left_heel = detect_joint(landmarks, 'LEFT_HEEL')
            right_heel = detect_joint(landmarks, 'RIGHT_HEEL')
            left_mouth = detect_joint(landmarks, 'MOUTH_LEFT')
            right_mouth = detect_joint(landmarks, 'MOUTH_RIGHT')
            mouth = [
                (left_mouth[0] + right_mouth[0]) / 2,
                (left_mouth[1] + right_mouth[1]) / 2
            ]

            # Calculate angle
            joint_angle = JointAngle(landmarks)

            Left_Leg_Angle = joint_angle.left_leg_angle(left_hip, left_knee, left_ankle)
            Right_Leg_Angle = joint_angle.right_leg_angle(right_hip, right_knee, right_ankle)
            Neck_Angle = joint_angle.neck_angle(mouth, shoulder, hip)
            Left_Arm_Angle = joint_angle.left_arm_angle(left_shoulder, left_elbow, left_wrist)
            Right_Arm_Angle = joint_angle.right_arm_angle(right_shoulder, right_elbow, right_wrist)
            Abdomen_Angle = joint_angle.abdomen_angle(shoulder, hip, knee)
            Back_Angle = joint_angle.back_angle(shoulder, hip)
            Internal_Angle = joint_angle.internal_angle(hip, left_heel, right_heel)

            Arm_Deviation = int(abs(Left_Arm_Angle - Right_Arm_Angle))

            # Curl counter logic
            if Arm_Deviation > 20:
                stage = 'Wrong'
                msg = 'Lift your '
                if Left_Arm_Angle > Right_Arm_Angle:
                    msg += 'left'
                else:
                    msg += 'right'
                msg += ' arm up: ' + str(Arm_Deviation) + ' degrees'
            else:
                msg = 'Good'
                if Left_Arm_Angle > 120 and Right_Arm_Angle > 120:
                    stage = "Down"
                if Left_Arm_Angle < 60 and Left_Arm_Angle < 60 and stage == "Down":
                    stage = "Up"
                    counter += 1

            if Internal_Angle < 12:
                stage = 'Wrong'
                if msg == "Good":
                    msg = "Widen your legs"
                else:
                    msg += " \nWiden your legs"

            # Curl velocity
            # if frameNo % 10 == 0:
            #     if frameNo > 0:
            #         velocities['frameNo'].append(frameNo) 
            #     angles['leftArm'].append(Left_Arm_Angle)
            #     angles['rightArm'].append(Right_Arm_Angle)
            #     if len(angles['leftArm']) == n:
            #         w = np.array(angles['leftArm']).astype(int)
            #         t = np.array([0, 1/3])
            #         angularVelocity = np.gradient(w, t)[0].astype(int)
            #         velocities['leftArm'].append(angularVelocity)
            #         angles['leftArm'].pop(0)
            #     if len(angles['rightArm']) == n:
            #         w = np.array(angles['rightArm']).astype(int)
            #         t = np.array([0, 1/3])
            #         angularVelocity = np.gradient(w, t)[0].astype(int)
            #         velocities['rightArm'].append(angularVelocity)
            #         angles['rightArm'].pop(0)

            if frameNo % 10 == 0: 
                angles['leftArm'].append(Left_Arm_Angle)
                angles['rightArm'].append(Right_Arm_Angle)
                if len(angles['leftArm']) == n:
                    w = np.array(angles['leftArm']).astype(int)
                    t = np.array([0, 1/3])
                    angularVelocity = np.gradient(w, t)[0].astype(int)
                    angles['leftArm'].pop(0)
                    if angularVelocity > velocities['leftArm'][frameNo / 10 -1]:
                        stage = 'Wrong'
                        if msg == 'Good':
                            msg = 'Left Arm - Slow down'
                        else:
                            msg += '\nLeft Arm - Slow down'
                    if angularVelocity < velocities['leftArm'][frameNo / 10 -1]:
                        stage = 'Wrong'
                        if msg == 'Good':
                            msg = 'Left Arm - Faster'
                        else:
                            msg += '\nLeft Arm - Faster'

                if len(angles['rightArm']) == n:
                    w = np.array(angles['rightArm']).astype(int)
                    t = np.array([0, 1/3])
                    angularVelocity = np.gradient(w, t)[0].astype(int)
                    angles['rightArm'].pop(0)
                    if angularVelocity > velocities['rightArm'][frameNo / 10 -1]:
                        stage = 'Wrong'
                        if msg == 'Good':
                            msg = 'Right Arm - Slow down'
                        else:
                            msg += '\nRight Arm - Slow downSlow down'
                    if angularVelocity < velocities['rightArm'][frameNo / 10 -1]:
                        stage = 'Wrong'
                        if msg == 'Good':
                            msg = 'Right Arm - Slow downFaster'
                        else:
                            msg += '\nRight Arm - Slow downFaster'               

            # Visualize
            cv2.putText(
                image,
                "Left Leg: " + str(int(Left_Leg_Angle)),
                tuple(np.multiply(
                    left_knee, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Right Leg: " + str(int(Right_Leg_Angle)),
                tuple(np.multiply(
                    right_knee, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Neck:" + str(int(Neck_Angle)),
                tuple(np.multiply(
                    neck, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Left Arm: " + str(int(Left_Arm_Angle)),
                tuple(np.multiply(
                    left_elbow, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255) if Arm_Deviation > 20 else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Right Arm: " + str(int(Right_Arm_Angle)),
                tuple(np.multiply(
                    right_elbow, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255) if Arm_Deviation > 20 else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Internal: " + str(int(Internal_Angle)),
                tuple(np.multiply(
                    [hip[0], hip[1] * 1.2], [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255) if Internal_Angle < 12 else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Abdomen: " + str(int(Abdomen_Angle)),
                tuple(np.multiply(
                    hip, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Back: " + str(int(Back_Angle)),
                tuple(np.multiply(
                    ref, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except:
            pass

        # Setup status box
        display_table(image, counter, stage, msg, angularVelocity)

        # Render detection
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("Posture Detection", image)

        frameNo += 1

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Write to file csv
    # df = pd.DataFrame(velocities)
    # df.to_csv('dataset/velocities.csv')

cap.release()
cv2.destroyAllWindows()