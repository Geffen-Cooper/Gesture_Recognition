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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Create RNN Model
class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        # self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)#.to('cuda')
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)#.to('cuda')
            
        # loop
        out, (hi,ci) = self.rnn(x[:,0,:].unsqueeze(1), (h0,c0))
        pred = self.fc(out[:,-1,:])
        for i in range(1,x.shape[1]):
            out, (hi,ci) = self.rnn(x[:,i,:].unsqueeze(1), (hi,ci))
            pred += self.fc(out[:,-1,:])
        return pred/x.shape[1]
    
model = RNNModel(63,256,1,25)
model.load_state_dict(torch.load("test_acc_50.pth"))
model.eval()

# initialize the capture object
cap = cv2.VideoCapture(0)
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

mpHands = mp.solutions.hands
hands = mpHands.Hands(model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils



count = 0

start = False
# start loop
while True:
    # try to read a frame
    ret,image = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    # image = cv2.flip(image,1)
    image = cv2.resize(image,(320, 240))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        if start == False:
            start = True
            print("STARTING===========")
            lms_x = ["lmx"+str(i) for i in range(21)]
            lms_y = ["lmy"+str(i) for i in range(21)]
            lms_z = ["lmz"+str(i) for i in range(21)]
            col_names = lms_x + lms_y + lms_z
            df = pd.DataFrame(columns=col_names)
        # print(results.multi_hand_landmarks[0].landmark[0])
        # exit()
        # print(f"count:{count} --> HAND")
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
    elif start == True:
        # print(f"count:{count} --> NOTHING")
        df.loc[len(df.index)] = [0 for j in range(63)]

    if count == 79:
        start = False
        landmarks_seq = df.values
        for col in range(landmarks_seq.shape[1]):
            x = landmarks_seq[:,col]
            x = medfilt(x,5)
            x = (x-np.min(x))/(np.max(x)-(np.min(x))+1e-6)
            x = (x-np.mean(x))/(np.std(x)+1e-6)
            landmarks_seq[:,col] = x


        landmarks_seq = transforms.ToTensor()(landmarks_seq).float()

        with torch.no_grad():
            model.eval()
            pred = model(landmarks_seq)

        print("prediction:",torch.argmax(pred).item())
        count = 0
    
    if start == True:
        count += 1

    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = "FPS: " + str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0
    # img, text, location of BLC, font, size, color, thickness, linetype
    cv2.putText(image, fps+", "+str(int(W))+"x"+str(int(H)), (7, 30), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('window',image)

    # quit when click 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break