import cv2
import numpy as np
import pdb
import pandas as pd

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def load_split_nvgesture(file_with_split = './nvgesture_train_correct.lst',list_split = list()):
    params_dictionary = dict()
    with open(file_with_split,'rb') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]

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
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
        
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            list_split.append(params_dictionary)
 
    return list_split

def load_data_from_file(example_config, sensor,image_width, image_height):

    path = example_config[sensor] + ".avi"
    start_frame = example_config[sensor+'_start']
    end_frame = example_config[sensor+'_end']
    label = example_config['label']

    frames_to_load = range(start_frame, end_frame)

    chnum = 3 if sensor == "color" else 1

    video_container = np.zeros((image_height, image_width, chnum, 80), dtype = np.uint8)

    cap = cv2.VideoCapture(path)

    ret = 1
    frNum = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);
    for indx, frameIndx in enumerate(frames_to_load):  
        if indx == 80:
            break  
        ret, frame = cap.read()
        # print(indx,frame.shape,frames_to_load)
        if ret:
            frame = cv2.resize(frame,(image_width, image_height))
            if sensor != "color":
                frame = frame[...,0]
                frame = frame[...,np.newaxis]
            video_container[..., indx] = frame
        else:
            print("Could not load frame")
    cap.release()

    return video_container, label

if __name__ == "__main__":
    sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
    file_lists = dict()
    file_lists["test"] = "./nvgesture_test_correct_cvpr2016.lst"
    file_lists["train"] = "./nvgesture_train_correct_cvpr2016.lst"
    train_list = list()
    test_list = list()

    load_split_nvgesture(file_with_split = file_lists["train"],list_split = train_list)
    load_split_nvgesture(file_with_split = file_lists["test"],list_split = test_list)

    lms_x = ["lmx"+str(i) for i in range(21)]
    lms_y = ["lmy"+str(i) for i in range(21)]
    lms_z = ["lmz"+str(i) for i in range(21)]
    col_names = lms_x + lms_y + lms_z

    def rotx(deg):
        return np.array([[1,0,0],
                         [0,np.cos(np.deg2rad(deg)),-np.sin(np.deg2rad(deg))],
                         [0,np.sin(np.deg2rad(deg)),np.cos(np.deg2rad(deg))]])
    def roty(deg):
        return np.array([[np.cos(np.deg2rad(deg)),0,np.sin(np.deg2rad(deg))],
                         [0,1,0],
                         [-np.sin(np.deg2rad(deg)),0,np.cos(np.deg2rad(deg))]])
    def rotz(deg):
        return np.array([[np.cos(np.deg2rad(deg)),-np.sin(np.deg2rad(deg)),0],
                         [np.sin(np.deg2rad(deg)),np.cos(np.deg2rad(deg)),0],
                         [0,0,1]])

    
    for sample_i in range(len(test_list)):
    # for sample_i in range(len(train_list)):
        df = pd.DataFrame(columns=col_names)
        # data, label = load_data_from_file(example_config = train_list[0], sensor = sensors[2], image_width = 320, image_height = 240)
        
        data, label = load_data_from_file(example_config = test_list[sample_i], sensor = sensors[2], image_width = 320, image_height = 240)
        # data, label = load_data_from_file(example_config = train_list[sample_i], sensor = sensors[2], image_width = 320, image_height = 240)
        data = np.transpose(data,(3,0,1,2))

        with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            for frame_i,image in enumerate(data):
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = np.stack((image[:,:,0],)*3, axis=-1)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # print(results.multi_hand_world_landmarks)
                # exit()
                if results.multi_hand_landmarks:
                    # print(results.multi_hand_landmarks[0].landmark[0])
                    # exit()
                    for hand_landmarks in results.multi_hand_landmarks:
                        # for lm_i,lm in enumerate(hand_landmarks.landmark):
                        #     if frame_i == 32 and (lm_i == 0 or lm_i == 8 or lm_i == 20):
                        #         print(f"before:{frame_i},{lm_i},{np.array([lm.x,lm.y,lm.z])}")
                        #         # print(rotx(-90))
                        #     new_coord = rotx(90)@np.array([lm.x,lm.y,lm.z])
                        #     hand_landmarks.landmark[lm_i].x = new_coord[0]
                        #     hand_landmarks.landmark[lm_i].y = new_coord[1]+.3
                        #     hand_landmarks.landmark[lm_i].z = new_coord[2]
                        #     if frame_i == 32 and (lm_i == 0 or lm_i == 8 or lm_i ==20):
                        #         print(f"after:{frame_i},{lm_i},{np.array([lm.x,lm.y,lm.z])}")
                        # exit()
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
                else:
                    df.loc[len(df.index)] = [0 for j in range(63)]
                        # print(hand_landmarks)
                    # print(i,len(results.multi_hand_landmarks[0].landmark))
                # Flip the image horizontally for a selfie-view display.
                # cv2.imwrite('frame_'+str(i)+".jpg",image)
                
                # print(frame.shape)
                # cv2.imwrite("frame_"+str(frame_i)+".jpg",image)
            # df.to_csv("_frame_"+str(sample_i)+"_label_"+str(label)+".csv",index=False)
            df.to_csv("lm_test_top/sample_"+str(sample_i)+"_label_"+str(label)+".csv",index=False)
            # exit()
            # df.to_csv("lm_test/sample_"+str(sample_i)+"_label_"+str(label)+".csv",index=False)
            print(f"sample:{sample_i} of {len(test_list)}")
    # pdb.set_trace()