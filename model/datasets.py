import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np
import re

class Gestures(Dataset):
    def __init__(self, root_dir,transform=None,train=True,test=False,subset=None):
        """
        Args:
            root_dir (string): directory where the csvs are
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # if using a subset, provide a map to the original class
        if subset != None:
            self.class_idx_map = {i : subset[i] for i in range(0, len(subset))}

        # get the class ids
        # self.class_id_list = sorted(os.listdir(os.path.join(self.root_dir,"Video_data")))

        # # map from class id to clas idx, e.g. class_01... --> 0, the idx is the label
        # self.class_idx_map = {self.class_id_list[i] : i for i in range(0, len(self.class_id_list))}

        # store imgs and labels
        self.img_paths = [] 
        self.labels = []

        if train == True:
            subdir = "lm_train"
        elif test == True:
            subdir = "lm_test"

        files = os.listdir(os.path.join(root_dir,subdir))
        files.sort(key=lambda f: int(f.split("_")[1]))
        bad_idxs = [136,181,182,226,240,262,272,302,363,376]
        for i,file in enumerate(files):
            # if i in bad_idxs:
            #     continue
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
        return landmarks_seq[0], label, idx

    def __len__(self):
        return len(self.img_paths)

    def visualize_batch(self,model=None):
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)

        # get the first batch
        (landmarks_seqs, labels,id) = next(iter(data_loader))

        if model != None:
            out = model(landmarks_seqs)
            pred = torch.argmax(out,dim=1)
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                if model != None:
                    text = self.class_id_list[labels[idx]]+"\n"+str(id[idx])+"\n"+str(self.class_id_list(pred[idx]))
                else:
                    text = self.class_id_list[labels[idx]]+"\n"+str(id[idx])#os.path.basename(self.img_paths[id[idx]])

                ax_array[i,j].imshow(torch.permute(landmarks_seqs[idx],(1,0)))

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()

class ConvertAngles(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        print("hi")
        return sample


def load_nvgesture(batch_size, rand_seed, root_dir,subset=None):

    tsfms = transforms.Compose([
        transforms.ToTensor()
        ]
    )

    if subset != None:
        dataset = Gestures(root_dir,tsfms,train=True,subset=subset)
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*.9)])
        test_set = Gestures(root_dir,tsfms,train=False,test=True,subset=subset)
    else:
        dataset = Gestures(root_dir,tsfms,train=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*.9)])
        test_set = Gestures(root_dir,tsfms,train=False,test=True)

    # create the data loaders
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    
    # return test_loader
    return (train_loader, val_loader, test_loader)

# if __name__ == '__main__':
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         ConvertAngles()
#     ]
# )
#     ds = Gestures(root_dir='../data/nvGesture_v1', train=True, transform=transform)