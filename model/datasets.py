import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np
import re
from scipy.spatial.transform import Rotation


class Gestures(Dataset):
    def __init__(self, root_dir, transform=None, train=True, test=False, subset=None):
        """
        Args:
            root_dir (string): directory where the csvs are
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # if using a subset, provide a map to the original class
        if subset != None:
            self.class_idx_map = {i: subset[i] for i in range(0, len(subset))}

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

        files = os.listdir(os.path.join(root_dir, subdir))
        files.sort(key=lambda f: int(f.split("_")[1]))
        bad_idxs = [136, 181, 182, 226, 240, 262, 272, 302, 363, 376]
        for i, file in enumerate(files):
            # if i in bad_idxs:
            #     continue
            if subset != None:
                l = int(file.split("_")[-1][:-4])
                if l in subset:
                    self.img_paths.append(os.path.join(root_dir, subdir, file))
                    self.labels.append(subset.index(l))
            else:
                self.img_paths.append(os.path.join(root_dir, subdir, file))
                self.labels.append(int(file.split("_")[-1][:-4]))

    def __getitem__(self, idx):
        # read the csv
        landmarks_seq = pd.read_csv(self.img_paths[idx])

        # apply transform
        if self.transform:
            landmarks_seq = self.transform(landmarks_seq).float()

        # get the label
        label = self.labels[idx]

        # return the sample (landmark sequence (tensor)), object class (int)
        return landmarks_seq, label, idx

    def __len__(self):
        return len(self.img_paths)

    def visualize_batch(self, model=None):
        batch_size = 64
        data_loader = DataLoader(self, batch_size, shuffle=True)

        # get the first batch
        (landmarks_seqs, labels, id) = next(iter(data_loader))

        if model != None:
            out = model(landmarks_seqs)
            pred = torch.argmax(out, dim=1)

        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8

        fig, ax_array = plt.subplots(rows, cols, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i * rows + j
                if model != None:
                    text = self.class_id_list[labels[idx]] + "\n" + str(id[idx]) + "\n" + str(self.class_id_list(pred[idx]))
                else:
                    text = self.class_id_list[labels[idx]] + "\n" + str(id[idx])  # os.path.basename(self.img_paths[id[idx]])

                ax_array[i, j].imshow(torch.permute(landmarks_seqs[idx], (1, 0)))

                ax_array[i, j].title.set_text(text)
                ax_array[i, j].set_xticks([])
                ax_array[i, j].set_yticks([])
        plt.show()


class DataframeToNumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample.values


class NormalizeAndFilter:
    def __init__(self, median_filter):
        self.median_filter = median_filter

    def __call__(self, sample):
        landmarks_seq = sample
        for col in range(landmarks_seq.shape[1]):
            x = landmarks_seq[:, col]
            if self.median_filter:
                x = medfilt(x, 5)
            x = (x - np.min(x)) / (np.max(x) - (np.min(x)) + 1e-6)
            x = (x - np.mean(x)) / (np.std(x) + 1e-6)
            landmarks_seq[:, col] = x
        return landmarks_seq


class RotateAngles(object):
    def __init__(self, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z, renormalize_origin=True):
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.trans_z = trans_z
        self.renormalize_origin = renormalize_origin

    def __call__(self, sample):
        rot = np.array([
            np.random.uniform(-self.rot_x, self.rot_x),
            np.random.uniform(-self.rot_y, self.rot_y),
            np.random.uniform(-self.rot_z, self.rot_z)
        ])
        trans = np.array([
            np.random.uniform(-self.trans_x, self.trans_x),
            np.random.uniform(-self.trans_y, self.trans_y),
            np.random.uniform(-self.trans_z, self.trans_z)
        ])
        return RotateAngles.apply_transform(df=sample, rot=rot, trans=trans, renormalize_origin=self.renormalize_origin)

    @staticmethod
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


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        return transforms.ToTensor()(sample)[0]


class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, label, idx = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, label, idx

    def __len__(self):
        return len(self.subset)


def load_nvgesture(batch_size, rand_seed, root_dir, median_filter, augment_angles, subset=None):
    tsfm_lst = []
    if augment_angles:
        tsfm_lst.append(RotateAngles(rot_x=np.pi / 8, rot_y=np.pi / 8, rot_z=np.pi / 8, trans_x=0.1, trans_y=0.1, trans_z=0.1, renormalize_origin=True))
    tsfm_lst.extend([
        DataframeToNumpy(),
        NormalizeAndFilter(median_filter=median_filter),
        ToTensor(),
    ])

    train_transforms = transforms.Compose(tsfm_lst)
    test_transforms = transforms.Compose([
        DataframeToNumpy(),
        NormalizeAndFilter(median_filter=median_filter),
        ToTensor(),
    ])

    if subset != None:
        dataset = Gestures(root_dir, None, train=True, subset=subset)
        train_subset, val_subset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * .9)])
        train_set = SubsetWrapper(train_subset, transform=train_transforms)
        val_set = SubsetWrapper(val_subset, transform=test_transforms)
        test_set = Gestures(root_dir, test_transforms, train=False, test=True, subset=subset)
    else:
        dataset = Gestures(root_dir, None, train=True)
        train_subset, val_subset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * .9)])
        train_set = SubsetWrapper(train_subset, transform=train_transforms)
        val_set = SubsetWrapper(val_subset, transform=test_transforms)
        test_set = Gestures(root_dir, test_transforms, train=False, test=True)

    # create the data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=2)

    # return test_loader
    return (train_loader, val_loader, test_loader)


if __name__ == '__main__':
    transform = transforms.Compose([
        RotateAngles(rot_x=np.pi / 8, rot_y=np.pi / 8, rot_z=np.pi / 8, trans_x=0.1, trans_y=0.1, trans_z=0.1, renormalize_origin=True),
        DataframeToNumpy(),
        NormalizeAndFilter(median_filter=True),
        ToTensor(),
    ]
    )
    ds = Gestures(root_dir='../csvs/ds_Lw_Sc_C1_V0_Ri', train=True, transform=transform)
    x = next(iter(ds))
    print(x)
    print("done")

    tr, te, va = load_nvgesture(batch_size=8, rand_seed=42, root_dir='../csvs/ds_Lw_Sc_C1_V0_Ri', median_filter=False, augment_angles=False, subset=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24))

    for X in tr:
        assert X is not None
    for X in te:
        assert X is not None
    for X in va:
        assert X is not None

    print("done")

