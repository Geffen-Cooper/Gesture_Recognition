import torch
from torchvision import transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np

import time
import cv2
import mediapipe as mp
import pickle
import copy


# media pipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

# gesture description strings
descriptions = ["Swipe Hand Left","Swipe Hand Right","Swipe Hand Up","Swipe Hand Down",\
                "Swipe Two Fingers Left","Swipe Two Fingers Right","Swipe Two Fingers Up","Swipe Two FIngers Down",\
                "Swipe Index Finger Down","Beckon With Hand","Expand Hand","Jazz Hand","One Finger Up","Two Fingers Up","THree Fingers Up",\
                "Lift Hand Up","Move Hand Down","Move Hand Forward","Beckon With Arm","TwoFingers Clockwise","Two Fingers CounterClockwise",
                "Two Fingers Forward","Close Hand","Thumbs Up","OK"]

# convert a csv to a media pipe landmark object for visualization
def csv_to_mp_landmark(file):
    # read the lanmark csv
    df = pd.read_csv(file)

    # get a mp landmark object
    with open('landmark.pkl', 'rb') as f:
        mp_landmarks = pickle.load(f)
    f.close()
    
    # create a list of mp landmark objects from csv
    lm_list = []
    # iterate over rows (time)
    for row in range(len(df)):
        landmarks = copy.deepcopy(mp_landmarks)
        # iterate over landmarks
        for idx,lm in enumerate(mp_landmarks.landmark):
            landmarks.landmark[idx].x = df['lmx'+str(idx)][row]
            landmarks.landmark[idx].y = df['lmy'+str(idx)][row]
            landmarks.landmark[idx].z = df['lmz'+str(idx)][row]

        lm_list.append(landmarks)
    
    return lm_list

# some global variables
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
class_num = 0
last_lm_list = None

blank = np.zeros((240,320,3),dtype=np.uint8)
seq_idx = 0
collected_first_item = False

files_collected = [os.listdir("data/class_"+str(i)) for i in range(len(os.listdir("data")))]
print(files_collected)

# start loop
while True:
    # get the landmark list of the current instance
    last_lm_list = csv_to_mp_landmark(os.path.join("data","class_"+str(class_num),files_collected[class_num][count]))

    # reset the screen to black
    blank[:,:,:] = 0

    # reset the sequence index
    if seq_idx >= len(last_lm_list):
        seq_idx = 0

    # draw the hand landmarks on the image
    mp_drawing.draw_landmarks(
        blank,
        last_lm_list[seq_idx],
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    
    # add some text
    cv2.putText(blank, "class: "+files_collected[class_num][-1].split('_')[1] +" ("+str(count+1)+"/"+str(len(files_collected[class_num]))+")",\
                 (0, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(blank, descriptions[class_num], (0, 40), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('last sequence',blank)
    seq_idx += 1

    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    elif k == ord('c'):
        class_num += 1
        if class_num == 25:
            class_num = 0
        print(f"Switched to Class {class_num}")
    elif k == ord('x'):
        class_num -= 1
        if class_num == -1:
            class_num = 24
        print(f"Switched to Class {class_num}")
    elif k == ord('d'):
        print(f"Deleting {files_collected[class_num][-1]}")
        os.remove(os.path.join("data","class_"+str(class_num),files_collected[class_num][-1]))
        files_collected[class_num] = files_collected[class_num][:-1]
        
        if len(files_collected[class_num]) == 0:
            collected_first_item = False
        else:
            print("viewing sample:",files_collected[class_num][-1])
    elif k == ord('n'):
        count += 1
        if count == len(files_collected[class_num]):
            count = 0
    elif k == ord('b'):
        count -= 1
        if count == -1:
            count = len(files_collected[class_num])-1
