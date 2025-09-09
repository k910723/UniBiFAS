from .unibifas import UniBiFAS_Model

def BuildModel(cfg):
    model = UniBiFAS_Model(cfg)
    return model

