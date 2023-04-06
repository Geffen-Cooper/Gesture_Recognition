import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import mediapipe as mp
from scipy.spatial.transform import Rotation
from mediapipe.framework.formats import landmark_pb2


def apply_transform(rot, trans, df, renormalize_origin):
    """
    Applies a 3D rotation and translation to a DataFrame of hand landmarks.

    Parameters
    ----------
    rot : list or tuple of 3 floats
        Rotation angles around x, y, and z axes in radians, specified in the 'sxyz' convention.
    trans : list or tuple of 3 floats
        Translation values along x, y, and z axes.
    df : pandas DataFrame
        DataFrame of hand landmarks, with columns in the format 'lmx{i}', 'lmy{i}', 'lmz{i}'
        for i in range(21), where each row represents a hand pose.

    Returns
    -------
    pandas DataFrame
        DataFrame of transformed hand landmarks, with the same format as the input DataFrame.
    """
    # FOR TESTING:
    # df.loc[0] = [i for i in range(63)]
    # df.loc[1] = [i for i in range(100, 163)]

    # Get the number of rows in the DataFrame
    num_rows = df.shape[0]

    # Reshape the DataFrame to a 3D numpy array
    x_cols = [col for col in df.columns if col.startswith('lmx')]
    y_cols = [col for col in df.columns if col.startswith('lmy')]
    z_cols = [col for col in df.columns if col.startswith('lmz')]
    x = df[x_cols].values
    y = df[y_cols].values
    z = df[z_cols].values
    arr = np.stack((x, y, z), axis=-1)

    # Translate the landmarks so that the wrist (lm0) is at the origin
    if renormalize_origin:
        wrist_idx = 0
        wrist_pos = arr[:, wrist_idx, :]
        arr = arr - wrist_pos.reshape(-1, 1, 3)

    # Create the rotation matrix
    r = Rotation.from_euler('xyz', rot)

    # Apply the rotation and translation to the array of hand landmarks
    arr_transformed = r.apply(arr.reshape(-1, 3)) + trans

    # Translate the landmarks back to their original position
    if renormalize_origin:
        arr_transformed = arr_transformed.reshape(num_rows, 21, 3) + wrist_pos.reshape(-1, 1, 3)

    # Reshape and reorder the transformed array
    arr_transformed = arr_transformed.transpose(0, 2, 1).reshape(num_rows, -1)

    # Reshape the transformed array back to a 2D DataFrame
    df_transformed = pd.DataFrame(arr_transformed, columns=df.columns)

    return df_transformed





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
    original_window_image = np.array(image)
    decoded_window_image = np.array(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

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

            # Create temp df
            lms_x = ["lmx" + str(i) for i in range(21)]
            lms_y = ["lmy" + str(i) for i in range(21)]
            lms_z = ["lmz" + str(i) for i in range(21)]
            col_names = lms_x + lms_y + lms_z
            temp_dataframe = pd.DataFrame(columns=col_names)
            temp_dataframe.loc[0] = lm_list_x + lm_list_y + lm_list_z

            # Apply transform
            temp_dataframe = apply_transform(np.array([0,0,np.pi/4]), np.array([0,0,0]), temp_dataframe, renormalize_origin=True)
            decoded_landmark_vector = temp_dataframe.loc[0].values

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

    # Draw frames on windows
    cv2.imshow('original', original_window_image)
    cv2.imshow('decoded', decoded_window_image)

    # quit when click 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if df is not None:
            print("Saving...")
            save_path = Path(record_out_dir) / Path(record_filename)
            df.to_csv(save_path, index=False)
        break

    # Handle recording
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