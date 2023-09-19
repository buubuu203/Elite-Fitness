import cv2
import mediapipe as mp
import numpy as np
from utilities import *
from joint_angles import JointAngle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("Exercise_Videos/male-barbell-curl-front.mp4")
# Curl counter variables
counter = 0
stage = None
msg = ''

Left_Leg_Angle = 0
Right_Leg_Angle = 0
Neck_Angle = 0
Left_Arm_Angle = 0
Right_Arm_Angle = 0
Abdomen_Angle = 0
Back_Angle = 0

# Setup Mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            # Calculate angle
            joint_angle = JointAngle(landmarks)

            Left_Leg_Angle = joint_angle.left_leg_angle()
            Right_Leg_Angle = joint_angle.right_leg_angle()
            Neck_Angle = joint_angle.neck_angle()
            Left_Arm_Angle = joint_angle.left_arm_angle()
            Right_Arm_Angle = joint_angle.right_arm_angle()
            Abdomen_Angle = joint_angle.abdomen_angle()
            Back_Angle = joint_angle.back_angle()

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
                if Left_Arm_Angle < 60 and Left_Arm_Angle < 60 and (stage == "Down" or stage == "Wrong"):
                    stage = "Up"
                    counter += 1

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
                (0, 0, 255) if stage == 'Wrong' else (255, 255, 255),
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
                (0, 0, 255) if stage == 'Wrong' else (255, 255, 255),
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
        cv2.rectangle(image, (0, 0), (600, 180), (245, 117, 16), -1)

        # Rep data
        display_table(image, counter, stage, msg)

        # Render detection
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("Posture Detection", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Write to file csv

cap.release()
cv2.destroyAllWindows()
