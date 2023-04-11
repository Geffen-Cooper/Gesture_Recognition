import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self, min_count):
        super(PrototypicalLoss, self).__init__()
        self.min_count = min_count

    def forward(self, input, target):
        return prototypical_loss(input, target, self.min_count)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(embeddings, target, min_count):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target = target.to('cpu')
    embeddings = embeddings.to('cpu')

    classes, counts = torch.unique(target, return_counts=True)
    filtered_classes = classes[(counts >= min_count).nonzero()].squeeze()

    # assuming n_query, n_target constants
    def generatePrototypeAndQueries(filtered_classes_idx):
        class_num = filtered_classes[filtered_classes_idx]
        target_eq_class = target.eq(class_num)

        n_in_class = target_eq_class.sum().item()

        n_support = n_in_class // 2
        n_query = n_in_class - n_support

        query_idxs_for_class = target_eq_class.nonzero()[n_support:]
        support_idxs_for_class = target_eq_class.nonzero()[:n_support].squeeze(1)

        query_samples_for_class = embeddings[query_idxs_for_class]
        query_samples_classes = target[query_idxs_for_class]
        query_samples_class_idxs = torch.Tensor([filtered_classes_idx for _ in range(n_query)])

        support_samples_for_class = embeddings[support_idxs_for_class]

        prototype_for_class = support_samples_for_class.mean(dim=0)

        return prototype_for_class, query_samples_for_class, query_samples_classes, query_samples_class_idxs

    prototypes_lst, query_samples_lst, query_samples_classes_lst, query_samples_class_idxs_lst = zip(*map(generatePrototypeAndQueries, range(len(filtered_classes))))
    prototypes = torch.stack(prototypes_lst)
    query_samples = torch.cat(query_samples_lst, dim=0).squeeze()
    query_samples_classes = torch.cat(query_samples_classes_lst).squeeze().long()
    query_samples_class_idxs = torch.cat(query_samples_class_idxs_lst).squeeze().long()

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1)

    loss_val = -log_p_y.gather(1, query_samples_class_idxs.unsqueeze(dim=1)).mean()
    acc_val = (log_p_y.argmax(dim=1) == query_samples_class_idxs).float().mean()

    return loss_val, acc_val
