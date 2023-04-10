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

# initialize the capture object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,240)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

# create a window
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

# measure FPS
last_frame_time = 0
curr_frame_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX



# some global variables
count = 0

hand_in_frame = False
num_frames_with_no_hand = 0

class_num = 0

last_lm_list = None

blank = np.zeros((240,320,3),dtype=np.uint8)
seq_idx = 0
collected_first_item = False

files_collected = [os.listdir("data/class_"+str(i)) for i in range(len(os.listdir("data")))]
print(files_collected)

# start loop
while True:
    # try to read a frame
    ret,image = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # resize
    image = cv2.resize(image,(320, 240))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # check if detected hand
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        # check if the hand was not previously in the frame
        if hand_in_frame == False:
            hand_in_frame = True
            print("\nHand Entered Frame===========")
            lms_x = ["lmx"+str(i) for i in range(21)]
            lms_y = ["lmy"+str(i) for i in range(21)]
            lms_z = ["lmz"+str(i) for i in range(21)]
            col_names = lms_x + lms_y + lms_z
            df = pd.DataFrame(columns=col_names)
        
        # collect the landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            lm_list_x = []
            lm_list_y = []
            lm_list_z = []

            for lm in hand_landmarks.landmark:
                lm_list_x.append(lm.x)
                lm_list_y.append(lm.y)
                lm_list_z.append(lm.z)

        df.loc[len(df.index)] = lm_list_x+lm_list_y+lm_list_z
        count += 1
    # if hand was already in the frame and now its not, add a zero row
    elif hand_in_frame == True:
        df.loc[len(df.index)] = [0 for j in range(63)]
        num_frames_with_no_hand += 1
        count += 1

    # if hand absent for 20 frames or reach max frames, then finish gesture
    if num_frames_with_no_hand == 20 or count == 80:
        print(f"End Sequence after {count} frames")
        
        hand_in_frame = False
        num_frames_with_no_hand = 0
        if count < 30:
            print("False Positive")
            count = 0
            continue
    
        fn = "class_"+str(class_num)+"_"+str(time.time())+".csv"
        files_collected[class_num].append(fn)

        print(f"class: {class_num}, collected {len(files_collected[class_num])}")

        df.to_csv(os.path.join("data","class_"+str(class_num),files_collected[class_num][-1]),index=False)
        print(f"saved to file {fn}")
        count = 0
        collected_first_item = True
        hold = time.time()

    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = "FPS: " + str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0
   
    # display text on screen 
    cv2.putText(image, fps, (80, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "class: "+files_collected[class_num][-1].split('_')[1],\
                 (0, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, descriptions[class_num], (0, 40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('window',image)

    # display last collected sample
    if not hand_in_frame and len(files_collected[class_num]) > 0:
        last_lm_list = csv_to_mp_landmark(os.path.join("data","class_"+str(class_num),files_collected[class_num][-1]))
        blank[:,:,:] = 0

        if seq_idx >= len(last_lm_list):
            seq_idx = 0

        mp_drawing.draw_landmarks(
            blank,
            last_lm_list[seq_idx],
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        cv2.putText(blank, "class: "+files_collected[class_num][-1].split('_')[1],\
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
        
