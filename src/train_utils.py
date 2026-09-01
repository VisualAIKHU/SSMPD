from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

import pickle
from collections import defaultdict

from model import SSD300, SSD300_3Way

def initialize_state(n_classes, train_conf):
    model = SSD300(n_classes=n_classes)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    return model, optimizer, optim_scheduler, train_conf.start_epoch, None

def load_state_from_checkpoint(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_with_new_optim(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    
    return model, optimizer, optim_scheduler, start_epoch, None
    
def load_state(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state(args.n_classes, train_conf)
    else:
        if train_conf.reset_optimizer:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_with_new_optim(train_conf, checkpoint)
        else:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint(train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state(config, student_checkpoint)
    teacher_state = load_state(config, teacher_checkpoint)

    return *student_state, *teacher_state

def initialize_state_3way(n_classes, train_conf):
    model = SSD300_3Way(n_classes=n_classes)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    return model, optimizer, optim_scheduler, train_conf.start_epoch, None

def load_state_from_checkpoint_3way(n_classes, train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_3way_with_new_optim(n_classes, train_conf, checkpoint):
    init_model, optimizer, optim_scheduler, train_conf.start_epoch, _ = initialize_state_3way(n_classes, train_conf)

    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    
    #             seach_name = pret_name.replace("pred_convs", name_pred)
    #             init_param = dict(init_model.named_parameters())[seach_name]
    #             init_param.data.copy_(pret_param.data)
    #     elif pret_name in dict(pret_model.named_parameters()):
    #         init_param = dict(init_model.named_parameters())[pret_name]
    #         init_param.data.copy_(pret_param.data)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_3way(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state_3way(args.n_classes, train_conf)
    else:
        if train_conf.reset_optimizer:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way_with_new_optim(args.n_classes, train_conf, checkpoint)
        else:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way(args.n_classes, train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher_3way(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state_3way(config, student_checkpoint)
    teacher_state = load_state_3way(config, teacher_checkpoint)

    return *student_state, *teacher_state

def create_dataloader(config, dataset_class, sample_mode = None, **kwargs):
    if kwargs["condition"] == "train":
        if sample_mode == "two":
            sample = "Labeled"
            dataset = dataset_class(config.args, sample = sample,**kwargs)
            L_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            sample = "Unlabeled"
            dataset = dataset_class(config.args, sample = sample, **kwargs)
            U_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            return dataset, L_loader, U_loader
        else: 
            dataset = dataset_class(config.args, **kwargs)
            loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
    else:
        dataset = dataset_class(config.args, **kwargs)
        test_batch_size = config.args["test"].eval_batch_size * torch.cuda.device_count()
        loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False,
                              num_workers=config.dataset.workers,
                              collate_fn=dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here
    return dataset, loader

def converter(originpath, changepath, wantname):
    # Loading the 90percents.txt file and creating a dictionary where keys are the index
    with open("./imageSets/" + originpath, 'r') as f:
        data_90 = {idx+1: line.strip() for idx, line in enumerate(f)}

    # Loading the test2.txt file
    with open(changepath, 'r') as f:
        data_test2 = f.readlines()

    # Replacing the first number of each line in test2.txt with corresponding line in 90percents.txt
    data_test2_new = []
    for line in data_test2:
        items = line.split(',')
        index = int(items[0])
        items[0] = data_90[index]
        data_test2_new.append(','.join(items))

    # Writing the new data into a new file
    with open(wantname, 'w') as f:
        for line in data_test2_new:
            f.write(line)

def soft_update(teacher_model, student_model, tau):
    """
    Soft update model parameters.
    θ_teacher = τ*θ_student + (1 - τ)*θ_teacher

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    :param tau: interpolation parameter (0.001 in your case)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(tau*student_param.data + (1.0-tau)*teacher_param.data)

def copy_student_to_teacher(teacher_model, student_model):
    """
    Copy student model to teacher model.
    θ_teacher = θ_student

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(student_param.data)

class EMAScheduler():
    def __init__(self, config):
        self.use_scheduler = config.ema.use_scheduler
        self.start_tau = config.ema.tau
        self.scheduling_start_epoch = config.ema.scheduling_start_epoch
        self.max_tau = config.ema.max_tau
        self.min_tau = config.ema.min_tau
        self.last_tau = config.ema.tau

    @staticmethod
    def calc_tau(epoch, tau):
        tau = tau
        return tau
    
    def get_tau(self, epoch):
        if not self.use_scheduler:
            return self.start_tau
        else:
            new_tau = EMAScheduler.calc_tau(epoch, self.last_tau)
            return new_tau
            """
            if new_tau > self.max_tau: 
                return self.max_tau
            elif new_tau < self.min_tau: 
                return self.min_tau
            else:
                self.last_tau = new_tau 
                return new_tau
            """

# def translate_coordinate(box, feature_w, feature_h, ori_w, ori_h, is_GT):
#     x, y, w, h = box
#     new_x = int(x * feature_w) if is_GT else int(x / ori_w * feature_w)
#     new_y = int(y * feature_h) if is_GT else int(y / ori_h * feature_h)
#     new_w = int(w * feature_w) if is_GT else int(w / ori_w * feature_w)
#     new_h = int(h * feature_h) if is_GT else int(h / ori_h * feature_h)

# def compute_gap_from_features(features, box, ori_w, ori_h, idx, is_GT):
#     gaps = []
#         feature = feature[idx]
#         _, feature_h, feature_w = feature.size()
#         x, y, w, h = translate_coordinate(box, feature_w, feature_h, ori_w, ori_h, is_GT)
#         obj = feature[:, y:y+h, x:x+w]
#             gap_obj = F.avg_pool2d(obj.unsqueeze(0), kernel_size=obj.size()[1:]).squeeze()
#             gaps.append(gap_obj)

def split_features(features, len_L):
    L_features = [feature[:len_L] for feature in features]
    U_features = [feature[len_L:] for feature in features]
    return L_features, U_features

def translate_coordinate(box, feature_w, feature_h):
    x1, y1, x2, y2 = box
    x1 = int(x1 * feature_w)
    y1 = int(y1 * feature_h)
    x2 = int(x2 * feature_w)
    y2 = int(y2 * feature_h)

    if x1 == x2:
        if x2 >= feature_w: x1 -= 1
        else: x2 += 1

    if y1 == y2:
        if y2 >= feature_h: y1 -= 1
        else: y2 += 1

    return x1, y1, x2, y2

def compute_gap_batch(features, box, img_idx):    
    bbox_gaps = []
    for feature in features:
        x1, y1, x2, y2 = translate_coordinate(box, feature.size(3), feature.size(2))
        if x2 - x1 <= 0 or y2 - y1 <= 0:  # skip degenerate boxes
            continue

        cropped_feature = feature[img_idx, :, y1:y2, x1:x2]
        if cropped_feature.size(1) > 0 and cropped_feature.size(2) > 0:
            gap = F.avg_pool2d(cropped_feature, kernel_size=cropped_feature.size()[1:]).view(feature.size(1))
            bbox_gaps.append(gap)

    if bbox_gaps:
        return torch.mean(torch.stack(bbox_gaps, dim=0), dim=0)
    return 

def calc_contrastive_loss(pos, neg, tau=0.1):
    """SC loss exactly as written in the paper, Eq.(6)-(8).

    L_sim = -(1/Np) * sum_i log( p_pos(p_i) / (p_pos(p_i) + p_neg) )
    with p_pos(p_i) = exp(d_i / tau) and p_neg = sum_j exp(d_j / tau) over negatives.
    """
    pos_term = torch.exp(pos / tau)                 # (Np,)
    neg_term = torch.sum(torch.exp(neg / tau))      # scalar
    return -torch.mean(torch.log(pos_term / (pos_term + neg_term)))


def calc_weight_by_GAPVector_distance(features, GT, PL, len_L, input_size, tau1, tau2):
    ori_h, ori_w = input_size
    L_features, U_features = split_features(features, len_L)
    GTs, PLs = [], []
    per_image_mean_gaps_GT = []
    for idx, boxes in enumerate(GT):
        mean_gaps = [compute_gap_batch(L_features, box, idx) for box in boxes if not all(box == 0)]
        mean_gaps = [gap for gap in mean_gaps if gap is not None]
        if mean_gaps:  # Check if mean_gaps is not empty
            GTs += mean_gaps
            per_image_mean_gaps_GT.append(torch.mean(torch.stack(mean_gaps), dim=0))

    per_image_mean_gaps_PL = []
    for idx, boxes in enumerate(PL):
        mean_gaps = [compute_gap_batch(U_features, box.numpy(), idx) for box in boxes if not all(box == 0)]
        mean_gaps = [gap for gap in mean_gaps if gap is not None]
        if mean_gaps:  # Check if mean_gaps is not empty
            PLs += mean_gaps
            per_image_mean_gaps_PL.append(torch.mean(torch.stack(mean_gaps), dim=0))

    if per_image_mean_gaps_GT and per_image_mean_gaps_PL:  # Check if both lists are not empty
        GT_vectors = torch.stack(GTs, dim=0)
        contrastive_loss = torch.tensor(0.0, requires_grad=True)  # ensure requires_grad=True
        # SC loss, paper Eq. (5)-(8): split the pseudo-labels by their max cosine
        # similarity to a ground-truth vector, then pull positives toward the GT.
        cos_pos, cos_neg = [], []
        for PL_vector in PLs:
            cos_sim = F.cosine_similarity(PL_vector.unsqueeze(0), GT_vectors, dim=1)
            max_cos = torch.max(cos_sim)
            if max_cos > tau1:
                cos_pos.append(max_cos.unsqueeze(0))   # confident positive
            elif max_cos < tau2:
                cos_neg.append(max_cos.unsqueeze(0))   # confident negative
            # tau2 <= max_cos <= tau1 -> uncertain group, ignored (paper III-C)

        if len(cos_pos) != 0 and len(cos_neg) != 0:
            pos = torch.cat(cos_pos)
            neg = torch.cat(cos_neg)
            contrastive_loss = contrastive_loss + calc_contrastive_loss(pos, neg)
        per_image_mean_gaps_GT = torch.mean(torch.stack(per_image_mean_gaps_GT), dim=0)
        per_image_mean_gaps_PL = torch.mean(torch.stack(per_image_mean_gaps_PL), dim=0)
        
        # PAA weight, paper Eq. (1): w = ln(e + t), with t the squared L2 distance
        # between the ground-truth and the pseudo-label GAP vectors.
        mse_norm = torch.sum((per_image_mean_gaps_GT - per_image_mean_gaps_PL) ** 2)   # squared L2
        weight = torch.log(math.e + mse_norm)

        return weight, contrastive_loss, mse_norm

    else:
        # Handle the case where one or both of the lists are empty
        return torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)  # Default weight with requires_grad=True

