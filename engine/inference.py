import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


from utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold, Find_Optimal_Cutoff

def do_eval(val_loader, model, device, log):
    """
        Eval the FAS model on the target data.
        The function evaluates the model on the target data and return TPR@FPR, HTER
        and AUC.
        Used in train.py to eval the model after each epoch; used in test.py to eval
        the final model.
    """

    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()

    prob_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for idx, (img, labels) in enumerate(val_loader):
            img = img.to(device)
            labels = labels.to(device)
            
            # Extract binary labels (first element of hierarchical labels)
            binary_labels = labels[:, 0]

            # Forward pass through the model
            # Model returns: img_feat_norm, patch_tokens, text_feat_b, text_feat_a, text_feat_art
            img_feat_norm, patch_tokens, text_feat_b, text_feat_a, text_feat_art = model(img)

            # Compute logits for binary classification
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat_norm @ text_feat_b.t()

            valid_loss = criterion(logits, binary_labels)
            valid_losses.update(valid_loss.item())

            acc = accuracy(logits, binary_labels, topk=(1,))
            valid_top1.update(acc[0].item())

            # Get probabilities for the positive class (spoof/fake)
            prob = F.softmax(logits, dim=1).cpu().data.numpy()[:, 1]
            label = binary_labels.cpu().data.numpy()

            prob_list = np.append(prob_list, prob)
            label_list = np.append(label_list, label)

    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    #cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    fpr, tpr, thr = roc_curve(label_list, prob_list)
    tpr_filtered = tpr[fpr <= 1 / 100]
    if len(tpr_filtered) == 0:
        rate = 0
    else:
        rate = tpr_filtered[-1]

    threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thr)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold_cs)
    
    #if log is not None:
    #    log.write(f"TPR@FPR = {rate}\n", is_file=True)
    #else:
    #    print("TPR@FPR = ", rate)

    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate]
