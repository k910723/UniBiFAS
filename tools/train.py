import torch

from modeling import BuildModel
from solver import make_optimizer, create_lr_scheduler
from data import BuildLoader
from engine.trainer import run
import losses

def train(cfg, log):

    # Load data
    train_loader = BuildLoader(cfg, isTrain = True, log = log)
    val_loader = BuildLoader(cfg, isTrain = False, log = log)

    # Instantiate the model
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    print('\n--- Loading Model ---')
    model = BuildModel(cfg).to(device)

    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created. Trainable parameters: {trainable_params / 1e6:.2f}M")
    #print("Trainable parameters should only be in the HierarchicalPromptLearner.")

    criterion = {
        'cls_b': getattr(torch.nn, cfg['losses']['cls_b']['name'])(**cfg['losses']['cls_b']['params']),
        'cls_a': getattr(torch.nn, cfg['losses']['cls_a']['name'])(**cfg['losses']['cls_a']['params']),
        'cls_art': getattr(torch.nn, cfg['losses']['cls_art']['name'])(**cfg['losses']['cls_art']['params']),
        'seg': getattr(torch.nn, cfg['losses']['seg']['name'])(**cfg['losses']['seg']['params'])
    }

    optimizer = make_optimizer(cfg, model, log)
    scheduler = create_lr_scheduler(optimizer, **cfg['scheduler']['params'])
    scaler = torch.amp.GradScaler('cuda') if cfg['train']['amp'] else None

    # Load if checkpoint is provided
    if cfg['ckpt']:
        ckpt = torch.load(cfg['ckpt'])
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        iter_num_start = epoch * cfg['iter_per_epoch']
        if cfg['train']['amp']:
            scaler.load_state_dict(ckpt["scaler"])
        log.write(f'\nLoaded checkpoint from epoch {epoch} at iteration : {iter_num_start}\n', is_file = 1)
    else:
        epoch = 1
        iter_num_start = 0
        log.write(f'\nStarting training from epoch {epoch} at iteration : {iter_num_start}\n', is_file = 1)


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
        epoch,
        iter_num_start
    )

    return hter, auc, tpr_fpr