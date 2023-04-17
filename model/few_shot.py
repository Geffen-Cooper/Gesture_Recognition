import torch
from torchvision import transforms
from datasets import *
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np
import torch.nn.functional as F
import pickle
from torch.utils.data import Subset

import time
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)


def update_centroids(model,data_path):
    eval_transforms = transforms.Compose([
        DataframeToNumpy(),
        NormalizeAndFilter(median_filter=False),
        ToTensor(),
    ])
    eval_dataset = Gestures(data_path, eval_transforms, train=True)

    # sampler to get an equal amount of each class
    few_shot_sampler = EvenSampler(eval_dataset)
    few_shot_sampler = EvenSampler(eval_dataset,shot=few_shot_sampler.min_len)
    train_idxs = [idx for idx in few_shot_sampler]
    train_batch_size = len(train_idxs)
    train_loader = DataLoader(Subset(eval_dataset,train_idxs), batch_size=train_batch_size, collate_fn=varying_length_collate_fn)
    train_batch = next(iter(train_loader))

    # print(train_batch)
    # print(train_batch[1],train_idxs)
    # print(eval_dataset.img_paths)
    # print(eval_dataset.labels)

    activation = {}
    # forward hook
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0].detach().to('cpu')
        return hook

    # register the forward hook as input to fully connected layer
    model.fc.register_forward_hook(get_activation('emb'))

    # forward pass on the batch to get embeddings
    with torch.no_grad():
        preds = model(train_batch[0])
        train_embds = activation['emb'].squeeze(1).to('cpu')
        # print(train_embds[:,:3],train_batch[1])

    # store centroids
    centroids = torch.zeros(len(train_batch[1].unique()),model.fc.in_features)
    for c in train_batch[1].unique():
        c_idxs = (train_batch[1] == c).nonzero()
        c_embds = train_embds[c_idxs]
        centroids[c] = c_embds.mean(dim=0)

    return centroids

def classify_sample(x,model,centroids):
    activation = {}
    # forward hook
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0].detach().to('cpu')
        return hook

    # register the forward hook as input to fully connected layer
    model.fc.register_forward_hook(get_activation('emb'))

    # forward pass on the batch to get embeddings
    with torch.no_grad():
        preds = model(x)
        embd = activation['emb'].squeeze(1).to('cpu')
        print(embd.view(-1)[:3])

    dists = torch.cdist(embd,centroids).view(-1)
    vals,idxs = torch.topk(1/dists,min(5,len(dists)))
    classification = F.softmax(vals,dim=0)
    return classification,idxs

# Create RNN Model with attention
class AttentionRNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(AttentionRNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.device = device

        # RNN
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # attention
        self.attention = torch.nn.Linear(hidden_dim, 1)

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # (batch_size,sequence length) since we want a weight for each hidden state in the sequence
        attention_weights = torch.zeros((x.shape[0],x.shape[1])).to(self.device)

        # output shape is (batch size, sequence length, feature dim)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        
        # for each time step, get the attention weight (do this over the batch)
        for i in range(x.shape[1]):
            attention_weights[:,i] = self.attention(output[:,i,:]).view(-1)
        attention_weights = F.softmax(attention_weights,dim=1)

        # apply attention weights for each element in batch
        attended = torch.zeros(output.shape[0],output.shape[2]).to(self.device)
        for i in range(x.shape[0]):
            attended[i,:] = attention_weights[i]@output[i,:,:]

        return self.fc(attended)
    
# load the model
model = AttentionRNNModel(63,256,1,25,'cpu')
model.load_state_dict(torch.load("models/collected.pth",map_location=torch.device('cpu'))['model_state_dict'])
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

count = 0
no_hand_count = 0

hand_in_frame = False

descriptions = []

collect = False
inference = False
new_class = False
inferred = False
class_num = 0
class_counts = []
centroids = torch.zeros(1,model.fc.in_features)

if os.path.isfile('metadata.pkl'):
    with open('metadata.pkl', 'rb') as inp:
        metadata = pickle.load(inp)
        descriptions = metadata['descriptions']
        centroids = metadata['centroids']
        class_counts = metadata['class_counts']

        # print(descriptions,centroids,class_counts)

# start loop
while True:
    if collect == True and len(descriptions) == 0:
        class_name = input("Enter a gesture name:")
        descriptions.append(class_name)
        class_counts.append(0)
    elif collect == True and new_class == True:
        new_class = False
        class_name = input("Enter a gesture name:")
        descriptions.append(class_name)
        class_counts.append(0)
        class_num = len(class_counts)-1
        new_centroids = torch.zeros(len(class_counts),model.fc.in_features)
        # print(new_centroids.shape,centroids.shape)
        new_centroids[:-1,:] = centroids[:,:]
        centroids = new_centroids
    
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    # image = cv2.flip(image,1)
    image = cv2.resize(img,(320, 240))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        if hand_in_frame == False:
            hand_in_frame = True
            print("STARTING===========")
            lms_x = ["lmx"+str(i) for i in range(21)]
            lms_y = ["lmy"+str(i) for i in range(21)]
            lms_z = ["lmz"+str(i) for i in range(21)]
            col_names = lms_x + lms_y + lms_z
            df = pd.DataFrame(columns=col_names)
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
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
    elif hand_in_frame == True:
        # print(f"count:{count} --> NOTHING")
        df.loc[len(df.index)] = [0 for j in range(63)]
        no_hand_count += 1

    if no_hand_count == 20 or count == 80:
        hand_in_frame = False

        if count < 30:
            print("False Positive")
            count = 0
            no_hand_count = 0
            continue

        if collect == True:
            print("collected")
            df.to_csv("few_shot_data\\lm_train\\sample_"+str(class_counts[class_num])+"_label_"+str(class_num)+".csv",index=False)
            class_counts[class_num] += 1

        if inference == True:
            landmarks_seq = df.values
            for col in range(landmarks_seq.shape[1]):
                x = landmarks_seq[:,col]
                # x = medfilt(x,5)
                x = (x-np.min(x))/(np.max(x)-(np.min(x))+1e-6)
                x = (x-np.mean(x))/(np.std(x)+1e-6)
                landmarks_seq[:,col] = x

            landmarks_seq = transforms.ToTensor()(landmarks_seq).float()

            print("prediction:")
            classification,idxs = classify_sample(landmarks_seq,model,centroids)
            # print(classification)
            for i,c in enumerate(classification):
                print(f"{descriptions[idxs[i]]}, {classification[i]}")
            inferred = True
        count = 0
        no_hand_count = 0
    
    if hand_in_frame == True:
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
    cv2.putText(img, fps+", "+str(int(W))+"x"+str(int(H)), (7, 30), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"collect:{collect} ({class_num}), inference:{inference}", (0, 70), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    if inferred == True:
        for i,c in enumerate(classification):
            if i == 0:
                cv2.putText(img, f"{i+1}. {descriptions[idxs[i]]} ({round(classification[i].item(),3)})", (0, 150+i*30), font, 0.6, (0, 125, 20), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, f"{i+1}. {descriptions[idxs[i]]} ({round(classification[i].item(),3)})", (0, 150+i*30), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('window',img)

    # quit when click 'q' on keyboard
    k = cv2.waitKey(1)

    if k == ord('q'):
        # print(descriptions,centroids,class_counts)
        centroids = update_centroids(model,"few_shot_data")
        metadata = {'descriptions':descriptions, 'centroids':centroids, 'class_counts':class_counts}
        with open('metadata.pkl', 'wb') as outp:
            pickle.dump(metadata, outp, pickle.HIGHEST_PROTOCOL)
        cv2.destroyAllWindows()
        break
    elif k == ord('c'):
        collect = True
        inference = False
    elif k == ord('i'):
        inference = True
        collect = False
        # calculate centroids
        centroids = update_centroids(model,"few_shot_data")

    elif k == ord('v'):
        if collect == True:
            class_num += 1
            if class_num == len(descriptions):
                class_num = 0
            print(f"Switched to Class {class_num}")
    elif k == ord('x'):
        if collect == True:
            class_num -= 1
            if class_num == -1:
                class_num = len(descriptions)-1
            print(f"Switched to Class {class_num}")
    elif k == ord('n'):
        new_class = True
    