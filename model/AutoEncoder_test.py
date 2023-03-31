import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from AutoEncoder import Autoencoder

model = Autoencoder()

model_save_path = 'models/ae_model.pt'
model.load_state_dict(torch.load(model_save_path))
model.eval()

# initialize the capture object
cap = cv2.VideoCapture(0)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# create a window
cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.namedWindow("decoded", cv2.WINDOW_NORMAL)

# measure FPS
last_frame_time = 0
curr_frame_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Load mediapipe handpose model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False)

# Record dataset
record_out_dir = '../data/ae_unlabled_ds'
record_filename = f'record.csv'
Path(record_out_dir).mkdir(exist_ok=True, parents=True)
recording = False
df = None


while True:
    # try to read a frame
    ret, image = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    # image = cv2.flip(image,1)
    image = cv2.resize(image, (320, 240))
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute Pose
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            original_window_image = np.array(image)

            mp_drawing.draw_landmarks(
                original_window_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Convert landmarks into vector
            lm_list_x = []
            lm_list_y = []
            lm_list_z = []
            for lm in hand_landmarks.landmark:
                lm_list_x.append(lm.x)
                lm_list_y.append(lm.y)
                lm_list_z.append(lm.z)

            if recording:
                df.loc[len(df.index)] = lm_list_x + lm_list_y + lm_list_z

            frame_landmark_vector = torch.Tensor([lm_list_x + lm_list_y + lm_list_z])

            # Compress into latent space, then decode
            # batch = torch.unsqueeze(frame_landmark_vector, 0)
            model.eval()
            with torch.no_grad():
                z_vec = model.encoder(frame_landmark_vector)
                output = model.decoder(z_vec)
            decoded_landmark_vector = torch.squeeze(output)

            # print(z_vec)

            # Convert decoded landmarks into mediapipe format
            decoded_landmarks = hand_landmarks
            for i, lm in enumerate(decoded_landmarks.landmark):
                decoded_landmarks.landmark[i].x = decoded_landmark_vector[i]
                decoded_landmarks.landmark[i].y = decoded_landmark_vector[i + 21]
                decoded_landmarks.landmark[i].z = decoded_landmark_vector[i + 42]

            # Draw decoded landmarks onto image
            decoded_window_image = np.array(image)

            mp_drawing.draw_landmarks(
                decoded_window_image,
                decoded_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = "FPS: " + str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0
    # img, text, location of BLC, font, size, color, thickness, linetype
    cv2.putText(original_window_image, fps + ", " + str(int(W)) + "x" + str(int(H)), (7, 30), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('original', original_window_image)
    cv2.imshow('decoded', decoded_window_image)

    # quit when click 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if df is not None:
            print("Saving...")
            save_path = Path(record_out_dir) / Path(record_filename)
            df.to_csv(save_path, index=False)
        break

    if (cv2.waitKey(1) & 0xFF == ord('r')):
        if not recording:
            print("RECORDING =========== ")
            if df is None:
                lms_x = ["lmx" + str(i) for i in range(21)]
                lms_y = ["lmy" + str(i) for i in range(21)]
                lms_z = ["lmz" + str(i) for i in range(21)]
                col_names = lms_x + lms_y + lms_z
                df = pd.DataFrame(columns=col_names)
            else:
                print("Resuming...")
            recording = True
        else:
            print("=========== STOP RECORDING")
            recording = False