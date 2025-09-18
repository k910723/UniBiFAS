import torch

def make_optimizer(cfg, model, log):
    lr = cfg['optimizer']['params']['lr']
    weight_decay = cfg['optimizer']['params']['weight_decay']

    params_optimize = [
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr, 'weight_decay': weight_decay},
    ]

    optimizer = getattr(torch.optim, cfg['optimizer']['name'])(params_optimize)
    return optimizer