import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from utilities import *

class JointAngle:

    def __init__(self, lm):
        self.lm = lm
        
        
    def left_leg_angle(self, left_hip, left_knee, left_ankle):
        return calculate_joint_angle(left_hip, left_knee, left_ankle)
        

    def right_leg_angle(self, right_hip, right_knee, right_ankle):
        return calculate_joint_angle(right_hip, right_knee, right_ankle)
        

    def neck_angle(self, mouth, shoulder, hip):
        return abs(180 - calculate_joint_angle(mouth, shoulder, hip))
        

    def left_arm_angle(self, left_shoulder, left_elbow, left_wrist):
        return calculate_joint_angle(left_shoulder, left_elbow, left_wrist)
        
    
    def right_arm_angle(self, right_shoulder, right_elbow, right_wrist):
        return calculate_joint_angle(right_shoulder, right_elbow, right_wrist)
        

    def abdomen_angle(self, shoulder, hip, knee):
        return calculate_joint_angle(shoulder, hip, knee)
        
        
    def back_angle(self, shoulder, hip):
        ref = [hip[0], shoulder[1]]

        return calculate_joint_angle(shoulder, hip, ref)
    

    def internal_angle(self, hip, left_heel, right_heel):
        return calculate_joint_angle(left_heel, hip, right_heel)