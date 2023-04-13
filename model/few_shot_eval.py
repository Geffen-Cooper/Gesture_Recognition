import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import logging
from datasets import *
from models import *
from torch.utils.data import Subset

# pass in a trained model, path to data directory, and number of samples
# to use for training per class, the rest will be used for evaluation
def eval_centroid_model(model,data_path,k_shot):
    # create the dataset
    eval_transforms = transforms.Compose([
        DataframeToNumpy(),
        NormalizeAndFilter(median_filter=False),
        ToTensor(),
    ])
    eval_dataset = Gestures(data_path, eval_transforms, train=True)

    # sampler to get an equal amount of each class
    few_shot_sampler = EvenSampler(eval_dataset,k_shot)

    # these idxs are ordered as c0_idx, c1_idx,...,cn_idx,c0_idx,...
    train_idxs = [idx for idx in few_shot_sampler]
    eval_idxs = few_shot_sampler.remain_class_idxs

    print(f"train/test ratio --> {k_shot}/{len(eval_idxs[0])}")

    # for i,c in enumerate(eval_idxs):
    #     print(f"num eval samples for class {i}: {len(c)}")
    eval_idxs = [item for sublist in eval_idxs for item in sublist]

    # create a dataloader for training and testing
    train_batch_size = len(train_idxs) # this will be k*num_classes
    eval_batch_size = len(eval_idxs)

    train_loader = DataLoader(Subset(eval_dataset,train_idxs), batch_size=train_batch_size, collate_fn=varying_length_collate_fn)
    eval_loader = DataLoader(Subset(eval_dataset,eval_idxs), batch_size=eval_batch_size, collate_fn=varying_length_collate_fn)

    train_batch = next(iter(train_loader))
    eval_batch = next(iter(eval_loader))

    # print((train_batch[1] == 0).float().sum())
    # print((eval_batch[1] == 0).float().sum())

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

        preds = model(eval_batch[0])
        eval_embds = activation['emb'].squeeze(1).to('cpu')

    # store centroids
    centroids = torch.zeros(len(train_batch[1].unique()),model.fc.in_features)
    for c in train_batch[1].unique():
        c_idxs = (train_batch[1] == c).nonzero()
        c_embds = train_embds[c_idxs]
        centroids[c] = c_embds.mean(dim=0)

    # torch.save(centroids,"centroids.pth")

    # compare eval set
    dists = torch.cdist(eval_embds,centroids)
    classifications = dists.argmin(dim=1)

    acc = (classifications == eval_batch[1]).float().mean()
    print(f"{len(train_batch[1].unique())}-way {k_shot}-shot accuracy: {acc}")


if __name__ == '__main__':
    model = AttentionRNNModel(63,256,1,25,'cpu')
    model.load_state_dict(torch.load("models/collected.pth",map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    eval_centroid_model(model,"../csvs/collected_data",5)
