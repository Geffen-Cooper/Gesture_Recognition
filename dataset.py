import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np

class Gestures(Dataset):
    def __init__(self, root_dir,transform=None,train=True,test=False,subset=None):#,LOPO=True,CV_10_all=False,CV_10_per=False,folds=None,participant=None):
        """
        Args:
            root_dir (string): directory where the csvs are
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # get the class ids
        self.class_id_list = sorted(os.listdir(os.path.join(self.root_dir,"Video_data")))

        # map from class id to clas idx, e.g. class_01... --> 0, the idx is the label
        self.class_idx_map = {self.class_id_list[i] : i for i in range(0, len(self.class_id_list))}

        # store imgs and labels
        self.img_paths = [] 
        self.labels = []

        if train == True:
            subdir = "lm_train"
        elif test == True:
            subdir = "lm_test"

        files = os.listdir(os.path.join(root_dir,subdir))
        for file in files:
            if subset != None:
                l = int(file.split("_")[-1][:-4])
                if l in subset:
                    self.img_paths.append(os.path.join(root_dir,subdir,file))
                    self.labels.append(subset.index(l))
            else:
                self.img_paths.append(os.path.join(root_dir,subdir,file))
                self.labels.append(int(file.split("_")[-1][:-4]))

    def __getitem__(self, idx):
        # read the csv
        landmarks_seq = pd.read_csv(self.img_paths[idx]).values

        # median filter
        for col in range(landmarks_seq.shape[1]):
            x = landmarks_seq[:,col]
            x = medfilt(x,5)
            x = (x-np.min(x))/(np.max(x)-(np.min(x))+1e-6)
            x = (x-np.mean(x))/(np.std(x)+1e-6)
            landmarks_seq[:,col] = x

        # apply transform
        if self.transform:
            landmarks_seq = self.transform(landmarks_seq).float()

        # get the label
        label = self.labels[idx]
            
        # return the sample (landmark sequence (tensor)), object class (int)
        return landmarks_seq[0], label

    def __len__(self):
        return len(self.img_paths)

    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        (landmarks_seqs, labels) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = self.class_id_list[labels[idx]]

                ax_array[i,j].imshow(torch.permute(landmarks_seqs[idx],(2,1,0)))

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()