from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def load_split_nvgesture(file_with_split='./nvgesture_train_correct.lst', list_split=list()):
    params_dictionary = dict()
    with open(file_with_split, 'rb') as f:
        dict_name = file_with_split[file_with_split.rfind('/') + 1:]
        dict_name = dict_name[:dict_name.find('_')]

        for line in f:
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


def create_handpose_dataset(output_dir, sensor, use_clahe, lm_type, video_mode,  create_train=True, create_test=True, nvgesture_path="../data/nvGesture_v1", train_list_path='../data/nvGesture_v1/nvgesture_train_correct_cvpr2016.lst', test_list_path='../data/nvGesture_v1/nvgesture_test_correct_cvpr2016.lst'):
    """
    ############## Example Params ##################
    NVGESTURE_PATH = "../data/nvGesture_v1/"
    TEST_LIST_PATH = "../data/nvGesture_v1/nvgesture_test_correct_cvpr2016.lst"
    TRAIN_LIST_PATH = "../data/nvGesture_v1/nvgesture_train_correct_cvpr2016.lst"
    SENSOR = "color"
    USE_CLAHE = True
    CREATE_TRAIN = True
    CREATE_TEST = True
    TYPE = "WORLD+"
    VIDEO_MODE = False
    OUTPUT_DIR = '../data/ds1'
    ########################################
    """
    ############## Params ##################
    NVGESTURE_PATH = nvgesture_path
    TEST_LIST_PATH = test_list_path
    TRAIN_LIST_PATH = train_list_path
    SENSOR = sensor
    USE_CLAHE = use_clahe
    CREATE_TRAIN = create_train
    CREATE_TEST = create_test
    TYPE = lm_type
    VIDEO_MODE = video_mode
    OUTPUT_DIR = output_dir
    ########################################
    sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
    if SENSOR not in set(sensors):
        raise NotImplementedError("Invalid argument")
    if TYPE not in {"WORLD", "WORLD+", "NORMAL"}:
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
        for sample_i in tqdm(range(len(lst)), desc="Sample"):
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
                        if len(results.multi_hand_landmarks) > 1:
                            raise NotImplementedError("More than one hand detected!")

                        # Select normal or world landmarks
                        if TYPE == "NORMAL":
                            hand_landmarks = results.multi_hand_landmarks[0]
                            lm_list_x, lm_list_y, lm_list_z = [], [], []
                            TO_SKIP = {}
                        elif TYPE == "WORLD":
                            hand_landmarks = results.multi_hand_world_landmarks[0]
                            lm_list_x, lm_list_y, lm_list_z = [], [], []
                            TO_SKIP = {}
                        elif TYPE == "WORLD+":
                            hand_landmarks = results.multi_hand_world_landmarks[0]
                            wrist_lm = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
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
                        df.loc[len(df.index)] = [0 for j in range(63)]

                Path(out_path).mkdir(parents=True, exist_ok=True)
                df.to_csv(out_path / Path("sample_" + str(sample_i) + "_label_" + str(label) + ".csv"), index=False)

if __name__ == "__main__":
    create_handpose_dataset(output_dir="../data/ds1", sensor="color", use_clahe=True, lm_type="WORLD+", video_mode=False)