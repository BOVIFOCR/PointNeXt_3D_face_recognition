import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import math
from openpoints.utils import registry

LOSS = registry.Registry('loss')
LOSS.register_module(name='CrossEntropy', module=CrossEntropyLoss)
LOSS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSS.register_module(name='BCEWithLogitsLoss', module=BCEWithLogitsLoss)


@LOSS.register_module()
class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        logits = logits.clone()
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        # return logits
        gt = labels.contiguous().view(-1)
        loss = torch.nn.CrossEntropyLoss()(logits, gt)
        return loss
        # if self.return_valid:
        #     return loss, logits, gt
        # else:
        #     return loss


@LOSS.register_module()
class ArcFace(torch.nn.Module):
    def __init__(self, margin=0.5,
                 scale=32.0,
                 num_classes=None,
                 weight=None,
                 return_valid=False
                 ):
        super(ArcFace, self).__init__()
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes
        self.return_valid = return_valid

        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None

    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)

        cosine = pred

        one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
        mask = one_hot

        # cosine = F.linear(F.normalize(pred), F.normalize(self.weight))
        cosine_of_target_classes = cosine[mask == 1] # (None, )

        eps = 1e-6  # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        modified_cosine_of_target_classes = torch.cos(angles + self.margin)
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1) # (None,1)
        logits = cosine + (mask * diff) # (None, n_classes)
        logits = logits * self.scale
        loss = torch.nn.CrossEntropyLoss()(logits, gt)

        if self.return_valid:
            return loss, pred, gt
        else:
            return loss


@LOSS.register_module()
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss


@LOSS.register_module()
class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2):
        super(MaskedCrossEntropy, self).__init__()
        self.creterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logit, target, mask):
        logit = logit.transpose(1, 2).reshape(-1, logit.shape[1])
        target = target.flatten()
        mask = mask.flatten()
        idx = mask == 1
        loss = self.creterion(logit[idx], target[idx])
        return loss

@LOSS.register_module()
class BCELogits(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = BCEWithLogitsLoss(**kwargs)
        
    def forward(self, logits, targets):
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        targets = targets.contiguous().view(-1)
        num_clsses = logits.shape[-1]
        targets_onehot = F.one_hot(targets, num_classes=num_clsses).to(device=logits.device,dtype=logits.dtype)
        return self.criterion(logits, targets_onehot)

@LOSS.register_module()
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logit, target):
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)  # N,C,H,W => N,C,H*W
            logit = logit.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logit = logit.contiguous().view(-1, logit.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logit)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != logit.data.type():
                self.alpha = self.alpha.type_as(logit.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()




@LOSS.register_module()
class Poly1CrossEntropyLoss(torch.nn.Module):
    def __init__(self,
                 num_classes: int =50,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        labels = labels.contiguous().view(-1)

        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


@LOSS.register_module()
class Poly1FocalLoss(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

@LOSS.register_module()
class MultiShapeCrossEntropy(torch.nn.Module):
    def __init__(self, criterion_args, **kwargs):
        super(MultiShapeCrossEntropy, self).__init__()
        self.criterion = build_criterion_from_cfg(criterion_args)

    def forward(self, logits_all_shapes, points_labels, shape_labels):
        batch_size = shape_labels.shape[0]
        losses = 0
        for i in range(batch_size):
            sl = shape_labels[i]
            logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
            pl = torch.unsqueeze(points_labels[i], 0)
            loss = self.criterion(logits, pl)
            losses += loss
        return losses / batch_size

def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return LOSS.build(cfg, **kwargs)