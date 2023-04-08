import itertools
import os
import random
import string
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import multiprocessing
import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def load_split_nvgesture(file_with_split='./nvgesture_train_correct.lst'):
    list_split = []

    params_dictionary = dict()
    with open(file_with_split, 'rb') as f:
        dict_name = file_with_split[file_with_split.rfind('/') + 1:]
        dict_name = dict_name[:dict_name.find('_')]

        for idx, line in enumerate(f):
            params = line.decode().split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                parsed = param.split(':')
                key = parsed[0]
                if key == 'label':
                    # make label start from 0
                    label = int(parsed[1]) - 1
                    params_dictionary['label'] = label
                elif key in ('depth', 'color', 'duo_left'):
                    # othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                    sensor_name = key
                    # first store path
                    params_dictionary[key] = path + '/' + parsed[1]
                    # store start frame
                    params_dictionary[key + '_start'] = int(parsed[2])

                    params_dictionary[key + '_end'] = int(parsed[3])

            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']

            list_split.append(params_dictionary)

    return list_split


def load_video_as_array(example_config, sensor, image_width, image_height, nvGesture_path):
    """
    Loads videos as array of (image_height, image_width, chnum, 80), also returns label
    """
    path_according_to_config = example_config[sensor] + ".avi"
    path = Path(nvGesture_path) / Path(path_according_to_config)
    start_frame = example_config[sensor + '_start']
    end_frame = example_config[sensor + '_end']
    label = example_config['label']

    frames_to_load = range(start_frame, end_frame)

    chnum = 3 if sensor == "color" else 1

    video_container = np.zeros((image_height, image_width, chnum, 80), dtype=np.uint8)

    cap = cv2.VideoCapture(str(path))

    frNum = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);
    for indx, frameIndx in enumerate(frames_to_load):
        if indx >= 80:
            break
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (image_width, image_height))
            if sensor != "color":
                frame = frame[..., 0]
                frame = frame[..., np.newaxis]
            video_container[..., indx] = frame
        else:
            print("Could not load frame")
    cap.release()

    video_container = np.transpose(video_container, (3, 0, 1, 2))
    return video_container, label

def equalize_hist(frame, clahe):
    if frame.shape[-1] == 3:
        # converting to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame
    else:
        raise NotImplementedError("Frame should be 3-channel")


def create_handpose_dataset(params):
    """
    ############## Example Params ##################
    params = {
        'nvgesture_path': "../data/nvGesture_v1/",
        'test_list_path': "../data/nvGesture_v1/nvgesture_test_correct_cvpr2016.lst",
        'train_list_path': "../data/nvGesture_v1/nvgesture_train_correct_cvpr2016.lst",
        'sensor': "color",
        'use_clahe': True,
        'create_train': True,
        'create_test': True,
        'lm_type': "WORLD+",
        'video_mode': False,
        'output_dir': '../data/ds1',
        'resolution_method': "INTERPOLATE"
    }
    ########################################
    """
    ############## Params ##################
    NVGESTURE_PATH = params['nvgesture_path']
    TEST_LIST_PATH = params['test_list_path']
    TRAIN_LIST_PATH = params['train_list_path']
    PROGRESS_FILE = params['progress_file']
    SENSOR = params['sensor']
    USE_CLAHE = params['use_clahe']
    CREATE_TRAIN = params.get('create_train', True)
    CREATE_TEST = params.get('create_test', True)
    TYPE = params['lm_type']
    VIDEO_MODE = params['video_mode']
    OUTPUT_DIR = params['output_dir']
    RESOLUTION_METHOD = params['resolution_method']
    ########################################

    sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
    if SENSOR not in set(sensors):
        raise NotImplementedError("Invalid argument")
    if TYPE not in {"WORLD", "WORLD+", "NORMAL"}:
        raise NotImplementedError("Invalid Argument")
    if RESOLUTION_METHOD not in {"INTERPOLATE", "FIRST", "ZERO"}:
        raise NotImplementedError("Invalid Argument")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    train_list = load_split_nvgesture(file_with_split=TRAIN_LIST_PATH)
    test_list = load_split_nvgesture(file_with_split=TEST_LIST_PATH)

    lms_x = ["lmx" + str(i) for i in range(21)]
    lms_y = ["lmy" + str(i) for i in range(21)]
    lms_z = ["lmz" + str(i) for i in range(21)]
    col_names = lms_x + lms_y + lms_z

    lsts_to_run = []
    output_dirs = []
    if CREATE_TRAIN:
        lsts_to_run.append(train_list)
        output_dirs.append(Path(OUTPUT_DIR) / Path('lm_train/'))
    if CREATE_TEST:
        lsts_to_run.append(test_list)
        output_dirs.append(Path(OUTPUT_DIR) / Path('lm_test/'))

    for lst, out_path in zip(lsts_to_run, output_dirs):
        for sample_i in tqdm(range(len(lst)), desc=f"{Path(OUTPUT_DIR).stem}/{Path(out_path).stem}"):
            df = pd.DataFrame(columns=col_names)
            video, label = load_video_as_array(example_config=lst[sample_i], sensor=SENSOR, image_width=320, image_height=240, nvGesture_path=NVGESTURE_PATH)

            with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=not VIDEO_MODE) as hands:

                for frame_i, image in enumerate(video):

                    image.flags.writeable = True

                    # Convert to RGB
                    if image.shape[-1] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        # image = np.stack((image[:, :, 0],) * 3, axis=-1)
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                    # Apply CLAHE
                    if USE_CLAHE:
                        image = equalize_hist(frame=image, clahe=clahe)

                    # To improve performance, optionally mark the image as not writeable to pass by reference.
                    image.flags.writeable = False

                    # Predict Pose
                    results = hands.process(image)

                    if results.multi_hand_landmarks:

                        # Should only be one hand
                        hand_to_use = 0
                        if len(results.multi_hand_landmarks) > 1:
                            print(f"More than one hand in frame: {frame_i} for {lst[sample_i]}")

                            if RESOLUTION_METHOD == "FIRST":
                                hand_to_use = 0
                            elif RESOLUTION_METHOD == "INTERPOLATE":
                                hand_to_use = -1
                            elif RESOLUTION_METHOD == "ZERO":
                                hand_to_use = -1
                            else:
                                raise NotImplementedError()

                        if hand_to_use >= 0:
                            # Select normal or world landmarks
                            if TYPE == "NORMAL":
                                hand_landmarks = results.multi_hand_landmarks[hand_to_use]
                                lm_list_x, lm_list_y, lm_list_z = [], [], []
                                TO_SKIP = {}
                            elif TYPE == "WORLD":
                                hand_landmarks = results.multi_hand_world_landmarks[hand_to_use]
                                lm_list_x, lm_list_y, lm_list_z = [], [], []
                                TO_SKIP = {}
                            elif TYPE == "WORLD+":
                                hand_landmarks = results.multi_hand_world_landmarks[hand_to_use]
                                wrist_lm = results.multi_hand_landmarks[hand_to_use].landmark[mp_hands.HandLandmark.WRIST]
                                lm_list_x, lm_list_y, lm_list_z = [wrist_lm.x], [wrist_lm.y], [wrist_lm.z]
                                TO_SKIP = {mp_hands.HandLandmark.PINKY_DIP}


                            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                                if lm_idx in TO_SKIP:
                                    continue
                                lm_list_x.append(lm.x)
                                lm_list_y.append(lm.y)
                                lm_list_z.append(lm.z)
                            df.loc[len(df.index)] = lm_list_x + lm_list_y + lm_list_z
                        else:
                            if RESOLUTION_METHOD == "FIRST":
                                raise Exception("Shouldn't get here")
                            elif RESOLUTION_METHOD == "INTERPOLATE":
                                df.loc[len(df.index)] = [float('nan') for j in range(63)]
                            elif RESOLUTION_METHOD == "ZERO":
                                df.loc[len(df.index)] = [0 for j in range(63)]
                            else:
                                raise NotImplementedError()
                    else:
                        df.loc[len(df.index)] = [0 for j in range(63)]

                # Fill NA
                if RESOLUTION_METHOD == "INTERPOLATE":
                    df = df.interpolate(method='linear')

                Path(out_path).mkdir(parents=True, exist_ok=True)
                df.to_csv(out_path / Path("sample_" + str(sample_i) + "_label_" + str(label) + ".csv"), index=False)

    # Save Progress
    if Path(PROGRESS_FILE).is_file():
        with open(PROGRESS_FILE, 'r') as f:
            finished_params = json.load(fp=f)
            finished_params.append(params)
    else:
        finished_params = [params]

    with open(PROGRESS_FILE, 'w') as f:
        json.dump(finished_params, fp=f)

if __name__ == "__main__":
    def sanitize_path_string(path_string):
        # sanitize the name to remove any invalid characters
        valid_chars = set("\\/-_.() %s%s" % (string.ascii_letters, string.digits))
        sanitized_name = "".join(c if c in valid_chars else "_" for c in path_string)

        # return the sanitized path with the sanitized name
        return sanitized_name


    ###################
    # Create datasets #
    ###################

    FORCE_OVERWRITE = False
    progress_file = '../csvs/progress.txt'

    lm_types = ["NORMAL", "WORLD+", "WORLD"]
    sensors = ["color", "duo_left"]
    use_clahes = [True, False]
    video_modes = [True, False]
    resolution_method = ["INTERPOLATE", "FIRST", "ZERO"]

    tasks_queue = []

    for i, combination in tqdm(enumerate(product(lm_types, sensors, use_clahes, video_modes, resolution_method)), desc="Dataset", unit="dataset"):
        tqdm.write(f"{i}, {dict(lm_type=combination[0], sensor=combination[1], use_clahe=combination[2],  video_mode=combination[3], resolution_method=combination[4])}")

        lm_type_short = combination[0].replace("WORLD+", "Wp").replace("WORLD", "W").replace("NORMAL", "N")
        lm_type_str = f"L{lm_type_short.lower()}"
        sensor_str = f"S{combination[1][0].lower()}"
        use_clahe_str = f"C{int(combination[2])}"
        video_mode_str = f"V{int(combination[3])}"
        resolution_method_str = f"R{combination[4][0].lower()}"

        out_dir = sanitize_path_string(f"../csvs/ds_{lm_type_str}_{sensor_str}_{use_clahe_str}_{video_mode_str}_{resolution_method_str}")
        task = dict(output_dir=out_dir, lm_type=combination[0], sensor=combination[1], use_clahe=combination[2],  video_mode=combination[3], resolution_method=combination[4], nvgesture_path="../data/nvGesture_v1/", test_list_path="../data/nvGesture_v1/nvgesture_test_correct_cvpr2016.lst", train_list_path="../data/nvGesture_v1/nvgesture_train_correct_cvpr2016.lst", progress_file=progress_file)

        if not FORCE_OVERWRITE:
            if Path(progress_file).is_file():
                with open(progress_file, 'r') as f:
                    finished_params = json.load(fp=f)
            else:
                finished_params = []

            if task in finished_params:
                print(f"Skipping {out_dir} since it is complete...")
                continue

        tasks_queue.append(task)

    # Parallelize the loop using multiprocessing
    with multiprocessing.Pool(processes=8) as p:
        p.map(create_handpose_dataset, tasks_queue)