import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os
from torchvision.transforms.functional import to_pil_image
from PIL import Image

class ContrastiveLoss(torch.nn.Module):
    """
    Implements the contrastive loss using high-level features for spoof/live
    and mid-level features for known attack types.
    """
    def __init__(self, temp=0.7, lambda_known=0.5, lambda_artifact=0.3):
        super().__init__()
        self.temp = temp
        self.lambda_known = lambda_known
        self.lambda_artifact = lambda_artifact


    def forward(self, f_v, f_v_low, f_v_mid, t_live, t_spoof, t_known_list, t_artifact_list, spoof_v_mid):
        """
        Calculates the loss based on the formula:
        L_con = -log(exp(sim(f_v, t_spoof)) / (exp(sim(f_v, t_live)) + exp(sim(f_v_mid, t_known))))
        
        Args:
            f_v (torch.Tensor): Image features from the image [B, D].
            t_live (torch.Tensor): Text features for "a photo of a real face" [1, D].
            t_spoof (torch.Tensor): Text features for "a photo of a spoof face" [1, D].
            t_known_list (list[torch.Tensor]): List of text features for each known attack type.
        """

        # Normalize features
        f_v = F.normalize(f_v, dim=1)
        f_v_low = F.normalize(f_v_low, dim=1)
        f_v_mid = F.normalize(f_v_mid, dim=1)
        t_live = F.normalize(t_live, dim=1)
        t_spoof = F.normalize(t_spoof, dim=1)
        t_known_list = [F.normalize(t_known, dim=1) for t_known in t_known_list]
        t_artifact_list = [F.normalize(t_artifact, dim=1) for t_artifact in t_artifact_list]
        spoof_v_mid = F.normalize(spoof_v_mid, dim=1)

        # Compute cosine similarities
        sim_live = torch.matmul(f_v, t_live.t())  / self.temp # [B, 1]
        sim_spoof = torch.matmul(f_v, t_spoof.t()) / self.temp # [B, 1]
        # sim_live_mid = torch.matmul(f_v_mid, t_known_list[0].t()) / self.temp  # [B, 1]
        sim_known = torch.cat([torch.matmul(f_v_mid, t_known.t()) for t_known in t_known_list], dim=1)  # [B, K]
        sim_artifact = torch.cat([torch.matmul(f_v_low, t_artifact.t()) for t_artifact in t_artifact_list], dim=1)  # [B, K]
        
        sim_spoof_mid = torch.matmul(f_v_mid, spoof_v_mid.t()) / self.temp  # [B, 1]

        # Compute contrastive loss
        exp_sim_spoof = torch.exp(sim_spoof)
        exp_sim_live = torch.exp(sim_live)
        exp_sim_known = torch.exp(sim_known).sum(dim=1, keepdim=True)  # Sum over known attacks
        exp_sim_artifact = torch.exp(sim_artifact).sum(dim=1, keepdim=True)  # Sum over artifact types

        exp_sim_spoof_mid = torch.exp(sim_spoof_mid).sum(dim=1, keepdim=True)  # Sum over spoof types

        # loss = -torch.log(exp_sim_spoof / (exp_sim_live + exp_sim_known + 1e-8))  # [B, 1]
        loss_live = -torch.log(exp_sim_spoof / (exp_sim_live + 1e-8))  # [B, 1]
        loss_known = -torch.log(exp_sim_spoof / (exp_sim_known + 1e-8))  # [B, 1]
        loss_artifact = -torch.log(exp_sim_spoof / (exp_sim_artifact + 1e-8))  # [B, 1]
        
        loss_spoof_v_mid = -torch.log(exp_sim_spoof / (exp_sim_spoof_mid + 1e-8))  # [B, 1]

        # loss = loss_live + self.lambda_known * loss_known + self.lambda_artifact * loss_artifact  # [B, 1]
        loss = loss_live + self.lambda_artifact * loss_artifact + self.lambda_known * loss_known + self.lambda_artifact * loss_artifact + self.lambda_known * loss_spoof_v_mid  # [B, 1]
        return loss.mean()


def run_visual_prompt(cfg, model, train_loader, visual_prompt, optimizer, scaler, device, log):
    
    log.write("\n--- Starting Visual Prompt Tuning ---\n", is_file=1)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # visual_prompt.to(device)
    criterion = ContrastiveLoss(temp=cfg['train_visual_prompt']['temp'], lambda_known=cfg['train_visual_prompt']['lambda_known'], lambda_artifact=cfg['train_visual_prompt']['lambda_artifact'])

    mean = torch.tensor(cfg['transforms']['mean'], dtype=torch.float32).view(3,1,1)
    std = torch.tensor(cfg['transforms']['std'], dtype=torch.float32).view(3,1,1)
    for epoch in range(1, cfg['train_visual_prompt']['epochs'] + 1):
        for batch_idx, (img, scm, labels) in enumerate(train_loader):
            img, scm, labels = img.to(device), scm.to(device), labels.to(device)
            attack_labels = labels[:, 1]
            live_indices = (attack_labels == 0).nonzero(as_tuple=True)[0]
            spoof_indices = (attack_labels != 0).nonzero(as_tuple=True)[0]

            to_pil = T.ToPILImage()
            # Save original images for visualization
            if epoch == cfg['train_visual_prompt']['epochs'] and batch_idx == 0:
                os.makedirs(f"{os.path.dirname(cfg['train_visual_prompt']['save_path'])}/visualization", exist_ok=True)
                for i in live_indices:
                    # The img is normalized with given mean and std
                    save_img = img[i].detach().cpu().clone()
                    save_img = save_img * std + mean
                    save_img = torch.clamp(save_img, 0., 1.)
                    pil_img = to_pil(save_img)
                    pil_img.save(os.path.join(f"{os.path.dirname(cfg['train_visual_prompt']['save_path'])}/visualization", f'original_img_{i}.png'))

            # --- Forward Pass ---
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                #attack_img = torch.clamp(img + visual_prompt, 0, 1)
                gamma = torch.rand(img.size(0), 1, 1, 1).to(device) * 0.5  # Random gamma in [0, 0.5]
                # attack_img = (1 - gamma) * img + gamma * visual_prompt
                # attack_img = img + gamma * visual_prompt
                attack_img = img + visual_prompt
                attack_img = torch.clamp(attack_img, -2, 2)
                
                # Save attack_img for visualization (for entire last batch)
                if epoch == cfg['train_visual_prompt']['epochs'] and batch_idx == 0:
                    to_pil = T.ToPILImage()
                    os.makedirs(f"{os.path.dirname(cfg['train_visual_prompt']['save_path'])}/visualization", exist_ok=True)
                    for i in live_indices:
                        save_img = attack_img[i].detach().cpu().clone()
                        save_img = save_img * std + mean
                        save_img = torch.clamp(save_img, 0., 1.)
                        pil_img = to_pil(save_img)
                        pil_img.save(os.path.join(f"{os.path.dirname(cfg['train_visual_prompt']['save_path'])}/visualization", f'attack_img_{i}.png'))
                
                if len(live_indices) == 0:
                    break  # No live samples in this batch

                outputs = model(attack_img[live_indices])
                img_feat_norm, cls_tokens, patch_tokens, text_feat_b, text_feat_a, text_feat_art = outputs

                spoof_outputs = model(img[spoof_indices])
                spoof_img_feat_norm, spoof_cls_tokens, spoof_patch_tokens, spoof_text_feat_b, spoof_text_feat_a, spoof_text_feat_art = spoof_outputs
                
                loss = criterion(
                    img_feat_norm,
                    cls_tokens[:,0,:], 
                    cls_tokens[:,1,:], 
                    text_feat_b[0].unsqueeze(0), 
                    text_feat_b[1].unsqueeze(0), 
                    [text_feat_a[i].unsqueeze(0) for i in range(0, text_feat_a.shape[0])], # 0 is live
                    [text_feat_art[i].unsqueeze(0) for i in range(0, text_feat_art.shape[0])],  # 0 is live
                    spoof_cls_tokens[:,1,:],
                )

            # --- Backward Pass ---
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
        log.write(f"Epoch [{epoch}/{cfg['train_visual_prompt']['epochs']}], Loss: {loss.item():.4f}\n", is_file=1)

    prompt_save_path = cfg['train_visual_prompt']['save_path']
    torch.save(visual_prompt, prompt_save_path)

    # Save visual prompt as an image for visualization
    to_pil = T.ToPILImage()
    prompt_img = visual_prompt.squeeze(0).cpu()
    prompt_img = (prompt_img - prompt_img.min()) / (prompt_img.max() - prompt_img.min() + 1e-8)  # Normalize to [0, 1]
    pil_img = to_pil(prompt_img)
    # Remove .pth from path
    prompt_save_path = os.path.dirname(prompt_save_path)
    pil_img.save(os.path.join(prompt_save_path, 'visual_prompt.png'))