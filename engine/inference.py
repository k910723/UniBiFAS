import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


from utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold, Find_Optimal_Cutoff

def do_eval(val_loader, model, device, log, inference_weights=None, cfg=None):
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
    
    # Check model mode to determine inference strategy
    model_mode = cfg['model'].get('mode', 'UniBiFAS') if cfg is not None else 'UniBiFAS'
    use_attack_inference = model_mode in ['UniBiFAS', 'UniBiFAS_TI']  # Full hierarchical
    use_artifact_inference = model_mode not in ['UniBiFAS_uni_IT', 'UniBiFAS_uni_TI']  # All except unidirectional variants

    model.eval()
    with torch.no_grad():
        for idx, (img, labels) in enumerate(val_loader):
            img = img.to(device)
            labels = labels.to(device)
            
            # Extract binary labels (first element of hierarchical labels)
            binary_labels = labels[:, 0]

            # Forward pass through the model
            # Model returns: img_feat_norm, patch_tokens, text_feat_b, text_feat_a, text_feat_art
            img_feat_norm, cls_tokens, patch_tokens, text_feat_b, text_feat_a, text_feat_art = model(img)

            # Compute logits for binary classification
            logit_scale = model.logit_scale.exp()
            
            # Compute similarity scores with all text features
            logits_binary = logit_scale * img_feat_norm @ text_feat_b.t()  # [batch, 2]
            
            # Determine which hierarchical features to use
            if use_attack_inference and use_artifact_inference:
                # Full hierarchical inference (UniBiFAS, UniBiFAS_TI)
                logits_attack = logit_scale * img_feat_norm @ text_feat_a.t()  # [batch, num_attacks]
                logits_artifact = logit_scale * img_feat_norm @ text_feat_art.t()  # [batch, num_artifacts]
                
                spoof_score_attack = torch.max(logits_attack, dim=1, keepdim=True)[0]  # [batch, 1]
                spoof_score_artifact = torch.max(logits_artifact, dim=1, keepdim=True)[0]  # [batch, 1]
                
                # Use provided weights or default values
                if inference_weights is not None:
                    w_binary = inference_weights['binary']
                    w_attack = inference_weights['attack']
                    w_artifact = inference_weights['artifact']
                else:
                    w_binary = 0.4
                    w_attack = 0.3
                    w_artifact = 0.3
                
                spoof_score_combined = (
                    w_binary * logits_binary[:, 1:2] +  # spoof score from binary text
                    w_attack * spoof_score_attack + 
                    w_artifact * spoof_score_artifact
                )  # [batch, 1]
                
                logits = torch.cat([logits_binary[:, 0:1], spoof_score_combined], dim=1)  # [batch, 2]
                
            elif use_artifact_inference:
                # Artifact only inference (UniBiFAS_bi_IT, UniBiFAS_bi_TI)
                logits_artifact = logit_scale * img_feat_norm @ text_feat_art.t()  # [batch, num_artifacts]
                spoof_score_artifact = torch.max(logits_artifact, dim=1, keepdim=True)[0]  # [batch, 1]
                
                # Use provided weights or default (binary + artifact only)
                if inference_weights is not None:
                    w_binary = inference_weights['binary']
                    w_artifact = inference_weights['artifact']
                else:
                    w_binary = 0.5
                    w_artifact = 0.5
                
                spoof_score_combined = (
                    w_binary * logits_binary[:, 1:2] +  # spoof score from binary text
                    w_artifact * spoof_score_artifact
                )  # [batch, 1]
                
                logits = torch.cat([logits_binary[:, 0:1], spoof_score_combined], dim=1)  # [batch, 2]
            else:
                # Binary only inference (UniBiFAS_uni_IT, UniBiFAS_uni_TI)
                logits = logits_binary  # [batch, 2]

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
