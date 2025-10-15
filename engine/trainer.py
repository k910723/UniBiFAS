import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from timeit import default_timer as timer

from .inference import do_eval
from utils import AverageMeter, time_to_str, accuracy, save_checkpoint

def run(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    criterion, # Expecting a dict of loss functions: {'cls_b', 'cls_a', 'cls_art', 'seg'}
    device,
    log,
    start_epoch
):
    # --- For Tracking Best score ---
    best_ACC = 0.0
    best_HTER = 1.0
    best_AUC = 0.0
    
    # --- Average Meters for losses ---
    loss_binary_meter = AverageMeter()
    loss_attack_meter = AverageMeter()
    loss_artifact_meter = AverageMeter()
    loss_seg_meter = AverageMeter()
    loss_total_meter = AverageMeter()
    binary_classifier_top1 = AverageMeter()

    log.write(f"{'-' * 55} [START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {'-' * 55}\n\n", is_file=True)
    log.write(f"{'** Starting UniBiFAS Model Training! **':^149}\n", is_file=True)

    # --- Logging Header ---
    message = (
        f"|{'epoch':^7}|"
        f"{' VALID ':-^24}|"
        f"{' Train ':-^65}|"
        f"{' Current Best ':-^24}|"
        f"{'time':^13}|\n"
    )
    log.write(message, is_file=True)
    message = (
        f"|{'':^7}|"
        f"{'loss':^6}{'top-1':^6}{'HTER':^6}{'AUC':^6}|"
        f"{'lr':^10}{'L_bin':^9}{'L_atk':^9}{'L_art':^9}{'L_seg':^9}{'L_total':^10}{'top-1':^9}|"
        f"{'top-1':^8}{'HTER':^8}{'AUC':^8}|"
        f"{'':^13}|\n"
    )
    log.write(message, is_file=True)
    log.write(f"|{'-' * 137}|\n", is_file=True)

    # --- Training Loop ---
    start = timer()
    
    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        model.train()
        
        # Reset meters for each epoch
        loss_binary_meter.reset()
        loss_attack_meter.reset()
        loss_artifact_meter.reset()
        loss_seg_meter.reset()
        loss_total_meter.reset()
        binary_classifier_top1.reset()
        
        # Iterate through the entire dataset (one epoch)
        for batch_idx, (img, scm, labels) in enumerate(train_loader):
            img, scm, labels = img.to(device), scm.to(device), labels.to(device)
            # Rescale scm to [0, 1]
            scm = (scm - scm.min()) / (scm.max() - scm.min() + 1e-8)
            
            # Unpack hierarchical labels
            binary_labels = labels[:, 0]
            attack_labels = labels[:, 1]
            artifact_labels = labels[:, 2]

            # --- Forward Pass ---
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                # ASSUMPTION: The model now returns a tuple of features
                outputs = model(img)
                img_feat_norm, cls_tokens, patch_tokens, text_feat_b, text_feat_a, text_feat_art = outputs
                
                # --- 1. Hierarchical Classification Loss Calculation ---
                logit_scale = model.logit_scale.exp()
                
                logits_binary = logit_scale * img_feat_norm @ text_feat_b.t()
                logits_attack = logit_scale * cls_tokens[:,1,:] @ text_feat_a.t()
                logits_artifact = logit_scale * cls_tokens[:,0,:] @ text_feat_art.t()

                loss_binary = criterion['cls_b'](logits_binary, binary_labels)
                loss_attack = criterion['cls_a'](logits_attack, attack_labels)
                loss_artifact = criterion['cls_art'](logits_artifact, artifact_labels)

                # --- 2. Segmentation Loss Calculation ---
                # Create zero maps for real images and use SCM for fake images
                fake_indices = (binary_labels == 1).nonzero(as_tuple=True)[0]
                loss_seg = torch.tensor(0.0, device=device)

                if len(patch_tokens) > 0:  # Process all images
                    # Assume the "spoof" text feature is at index 1
                    spoof_text_feat = text_feat_b[1].unsqueeze(0).unsqueeze(-1) # Shape: [1, D, 1]
                    real_text_feat = text_feat_b[0].unsqueeze(0).unsqueeze(-1) # Shape: [1, D, 1]
                    
                    # Normalize patch tokens to calculate cosine similarity
                    patch_tokens = patch_tokens / (patch_tokens.norm(dim=2, keepdim=True) + 1e-8)
                    
                    # Softmax over similarity to spoof and real text features
                    similarity_to_spoof = patch_tokens @ spoof_text_feat  # [N, num_patches, 1]
                    similarity_to_real = patch_tokens @ real_text_feat    # [N, num_patches, 1]
                    similarity_map = torch.cat([similarity_to_real, similarity_to_spoof], dim=-1)  # [N, num_patches, 2]
                    similarity_map = F.softmax(similarity_map, dim=-1)[:, :, 1:]  # Probability of being spoof

                    # Reshape to a 2D map (assuming 14x14 patches for ViT-B/16)
                    h = w = int(similarity_map.shape[1]**0.5)
                    pred_map = similarity_map.squeeze(-1).view(len(binary_labels), 1, h, w)
                    
                    # Upsample the predicted map to match the SCM size
                    pred_map_upsampled = F.interpolate(pred_map, size=scm.shape[-2:], mode='bilinear', align_corners=False)

                    # Create target maps: zero maps for real images, SCM for fake images
                    target_maps = torch.zeros_like(scm)
                    if len(fake_indices) > 0:
                        target_maps[fake_indices] = scm[fake_indices]

                    # Calculate segmentation loss for all images
                    loss_seg = criterion['seg'](pred_map_upsampled, target_maps)

                # --- 3. Total Loss ---
                total_loss = (cfg['loss_weights']['binary'] * loss_binary +
                              cfg['loss_weights']['attack'] * loss_attack +
                              cfg['loss_weights']['artifact'] * loss_artifact +
                              cfg['loss_weights']['segmentation'] * loss_seg)

            # --- Backward Pass ---
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # --- Update Meters & Logging ---
            lr = optimizer.param_groups[0]['lr']
            # Store weighted losses to match what's used in total loss
            weighted_loss_binary = cfg['loss_weights']['binary'] * loss_binary.item()
            weighted_loss_attack = cfg['loss_weights']['attack'] * loss_attack.item()
            weighted_loss_artifact = cfg['loss_weights']['artifact'] * loss_artifact.item()
            weighted_loss_seg = cfg['loss_weights']['segmentation'] * loss_seg.item()
            
            loss_binary_meter.update(weighted_loss_binary)
            loss_attack_meter.update(weighted_loss_attack)
            loss_artifact_meter.update(weighted_loss_artifact)
            loss_seg_meter.update(weighted_loss_seg)
            loss_total_meter.update(total_loss.item())

            acc = accuracy(logits_binary, binary_labels, topk=(1,))
            binary_classifier_top1.update(acc[0].item())

        # --- End of Epoch: Validation and Logging ---
        if epoch % cfg['train']['print_interval'] == 0 or epoch == cfg['train']['epochs']:
            # Run validation
            valid_args = do_eval(val_loader, model, device, log)

            is_best = valid_args[3] < best_HTER or (valid_args[3] == best_HTER and valid_args[4] > best_AUC)
            if is_best:
                best_HTER = valid_args[3]
                best_ACC = valid_args[6]
                best_AUC = valid_args[4]
                TPR_at_FPR = valid_args[-1]
            
            # --- Save checkpoint logic here ---
            # save_checkpoint(...)
            if is_best and cfg.get('save_ckpt'):
                save_list = [
                    epoch,
                    valid_args,
                    best_HTER,
                    best_ACC,
                    best_AUC,
                    valid_args[5]  # threshold
                ]
                save_checkpoint(save_list, is_best, model, optimizer, scheduler, filename=cfg['train']['save_path'])

            print('\r', end='', flush=True)
            message = (
                f"|{epoch:^7d}|"
                f"{valid_args[0]:^6.2f}{valid_args[6]:^6.2f}{valid_args[3]*100:^6.2f}{valid_args[4]*100:^6.2f}|"
                f"{lr:^10.6f}"
                f"{loss_binary_meter.avg:^9.3f}{loss_attack_meter.avg:^9.3f}{loss_artifact_meter.avg:^9.3f}"
                f"{loss_seg_meter.avg:^9.3f}{loss_total_meter.avg:^10.3f}{binary_classifier_top1.avg:^9.2f}|"
                f"{float(best_ACC):^8.2f}{float(best_HTER*100):^8.2f}{float(best_AUC*100):^8.2f}|"
                f"{time_to_str(timer() - start, 'sec'):^12}|\n"
            )
            log.write(message, is_file=True)
            
        # Update scheduler at the end of each epoch
        if scheduler is not None:
            scheduler.step()

    return best_HTER * 100.0, best_AUC * 100.0, TPR_at_FPR