import torch
import torch.nn as nn
from modeling import BuildModel
from solver import make_optimizer, create_lr_scheduler
from data import BuildLoader
from engine.trainer import run
from engine.trainer_visual_prompt import run_visual_prompt
import losses

def train(cfg, log):
    # Instantiate the model
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    print('\n--- Loading Model ---')
    model = BuildModel(cfg).to(device)

    if cfg.get('visual_prompt_mode') and cfg['visual_prompt_mode'] == 'train':
        print('\n--- Visual Prompt Tuning Stage 1 ---')
        train_loader = BuildLoader(cfg, isTrain=False, isTrainVisualPrompt=True, log=log)

        visual_prompt = torch.empty(1, 3, cfg['transforms']['imgSize'], cfg['transforms']['imgSize'], requires_grad=True).to(device)
        nn.init.normal_(visual_prompt, std=0.02)
        visual_prompt = nn.Parameter(visual_prompt)

        optimizer = torch.optim.AdamW([visual_prompt], lr=cfg['train_visual_prompt']['lr'], weight_decay=cfg['train_visual_prompt']['weight_decay'])
        scaler = torch.amp.GradScaler('cuda') if cfg['train']['amp'] else None

        ckpt = torch.load(f"{cfg['train']['save_path']}/{cfg['dataset']['source']}_{cfg['dataset']['target']}_best.pt")
        model.load_state_dict(ckpt['state_dict'])

        run_visual_prompt(
            cfg,
            model,
            train_loader,
            visual_prompt,
            optimizer,
            scaler,
            device,
            log
        )

        hter, auc, tpr_fpr = 0.0, 0.0, 0.0  # Dummy values

    elif cfg.get('visual_prompt_mode') and cfg['visual_prompt_mode'] == 'finetune':
        print('\n--- Fine-tuning Entire Model Stage 2 ---')
        # Load data
        train_loader = BuildLoader(cfg, isTrain=True, isFineTune=True, log=log)
        val_loader = BuildLoader(cfg, isTrain=False, log=log)

        # Check which parameters are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created. Trainable parameters: {trainable_params / 1e6:.2f}M")
        #print("Trainable parameters should only be in the HierarchicalPromptLearner.")

        criterion = {
            'cls_b': getattr(torch.nn, cfg['losses']['cls_b']['name'])(**cfg['losses']['cls_b'].get('params', {})),
            'cls_a': getattr(torch.nn, cfg['losses']['cls_a']['name'])(**cfg['losses']['cls_a'].get('params', {})),
            'cls_art': getattr(torch.nn, cfg['losses']['cls_art']['name'])(**cfg['losses']['cls_art'].get('params', {})),
            'seg': getattr(torch.nn, cfg['losses']['seg']['name'])(**cfg['losses']['seg'].get('params', {}))
        }
        
        # Add entropy loss if configured
        if 'entropy' in cfg['losses']:
            criterion['entropy'] = getattr(losses, cfg['losses']['entropy']['name'])(**cfg['losses']['entropy'].get('params', {}))

        optimizer = make_optimizer(cfg, model, log)
        scheduler = create_lr_scheduler(optimizer, **cfg['scheduler']['params'])
        scaler = torch.amp.GradScaler('cuda') if cfg['train']['amp'] else None

        # Load if checkpoint is provided
        ckpt = torch.load(f"{cfg['train']['save_path']}/{cfg['dataset']['source']}_{cfg['dataset']['target']}_best.pt")
        # model.load_state_dict(ckpt['state_dict'], strict=False)
        # 1. Load the state dictionary from the checkpoint
        pretrained_dict = ckpt['state_dict']

        # 2. Get the state dictionary of the new, larger model
        model_dict = model.state_dict()

        # 3. Identify the keys from the pretrained model that we want to keep.
        # We keep a key if it exists in the new model AND its shape is the same.
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # 4. Update the new model's dictionary with the weights we're keeping.
        model_dict.update(pretrained_dict)

        # 5. Load the updated, combined state dictionary into the new model.
        model.load_state_dict(model_dict)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = int(ckpt['epoch'] + 1)  # Continue from next epoch
        # if cfg['train']['amp']:
        #     scaler.load_state_dict(ckpt["scaler"])
        log.write(f'\nLoaded checkpoint from epoch {ckpt["epoch"]}, continuing from epoch {start_epoch}\n', is_file = 1)


        hter, auc, tpr_fpr = run(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            log,
            start_epoch
        )
    
    else:
        # Load data
        train_loader = BuildLoader(cfg, isTrain=True, log=log)
        val_loader = BuildLoader(cfg, isTrain=False, log=log)

        # Check which parameters are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created. Trainable parameters: {trainable_params / 1e6:.2f}M")
        #print("Trainable parameters should only be in the HierarchicalPromptLearner.")

        criterion = {
            'cls_b': getattr(torch.nn, cfg['losses']['cls_b']['name'])(**cfg['losses']['cls_b'].get('params', {})),
            'cls_a': getattr(torch.nn, cfg['losses']['cls_a']['name'])(**cfg['losses']['cls_a'].get('params', {})),
            'cls_art': getattr(torch.nn, cfg['losses']['cls_art']['name'])(**cfg['losses']['cls_art'].get('params', {})),
            'seg': getattr(torch.nn, cfg['losses']['seg']['name'])(**cfg['losses']['seg'].get('params', {}))
        }
        
        # Add entropy loss if configured
        if 'entropy' in cfg['losses']:
            criterion['entropy'] = getattr(losses, cfg['losses']['entropy']['name'])(**cfg['losses']['entropy'].get('params', {}))

        optimizer = make_optimizer(cfg, model, log)
        scheduler = create_lr_scheduler(optimizer, **cfg['scheduler']['params'])
        scaler = torch.amp.GradScaler('cuda') if cfg['train']['amp'] else None

        # Load if checkpoint is provided
        if cfg.get('ckpt') and not cfg.get('save_ckpt'):
            ckpt = torch.load(cfg['ckpt'])
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1  # Continue from next epoch
            if cfg['train']['amp']:
                scaler.load_state_dict(ckpt["scaler"])
            log.write(f'\nLoaded checkpoint from epoch {ckpt["epoch"]}, continuing from epoch {start_epoch}\n', is_file = 1)
        else:
            start_epoch = 1
            log.write(f'\nStarting training from epoch {start_epoch}\n', is_file = 1)


        hter, auc, tpr_fpr = run(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            log,
            start_epoch
        )

    return hter, auc, tpr_fpr