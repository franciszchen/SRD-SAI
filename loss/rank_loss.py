import torch
import torch.nn.functional as F

def pairwise_ranking_loss(preds, margin=0.05, size_average = True):
    """
        preds:
            list of scalar Tensor.
            Each value represent the probablity of each class
                e.g) class = 3
                    preds = [logits1[class], logits2[class]]
    """
    if len(preds) <= 1:
        return torch.zeros(1).cuda()
    else:
        losses = []
        for pred in preds:
            loss = [] # low, super, aux, ens
            loss.append((pred[0]-pred[1] + margin).clamp(min = 0)) # low->super
            loss.append((pred[1]-pred[3] + margin).clamp(min = 0)) # super->ensemble
            loss.append((pred[2]-pred[3] + margin).clamp(min = 0)) # aux->ensemble
            loss.append((pred[0]-pred[3] + 2*margin).clamp(min = 0)) # low->ens
            loss = torch.sum(torch.stack(loss))
            losses.append(loss)
        losses = torch.stack(losses)
        if size_average:
            losses = torch.mean(losses)
        else:
            losses = torch.sum(losses)
        return losses