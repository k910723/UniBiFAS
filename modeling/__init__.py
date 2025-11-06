from .unibifas import UniBiFAS_Model
from .unibifas_TI import UniBiFAS_Model as UniBiFAS_TI_Model
from .unibifas_bi_IT import UniBiFAS_Model as UniBiFAS_bi_IT_Model
from .unibifas_bi_TI import UniBiFAS_Model as UniBiFAS_bi_TI_Model
from .unibifas_uni_IT import UniBiFAS_Model as UniBiFAS_uni_IT_Model
from .unibifas_uni_TI import UniBiFAS_Model as UniBiFAS_uni_TI_Model

def BuildModel(cfg):
    """
    Build model based on the mode specified in config.
    
    Available modes:
    - 'UniBiFAS': Original V→T / V↔T / T→V guidance direction (with bidirectional)
    - 'UniBiFAS_TI': Inverted T→V / V↔T / V→T guidance direction (with bidirectional)
    - 'UniBiFAS_bi_IT': Simplified V→T / T→V without bidirectional middle layers
    - 'UniBiFAS_bi_TI': Inverted T→V / V→T without bidirectional middle layers
    - 'UniBiFAS_uni_IT': Unidirectional V→T only (no T→V, no bidirectional)
    - 'UniBiFAS_uni_TI': Unidirectional T→V only (no V→T, no bidirectional, inverted)
    """
    mode = cfg['model'].get('mode', 'UniBiFAS')
    
    if mode == 'UniBiFAS':
        model = UniBiFAS_Model(cfg)
    elif mode == 'UniBiFAS_TI':
        model = UniBiFAS_TI_Model(cfg)
    elif mode == 'UniBiFAS_bi_IT':
        model = UniBiFAS_bi_IT_Model(cfg)
    elif mode == 'UniBiFAS_bi_TI':
        model = UniBiFAS_bi_TI_Model(cfg)
    elif mode == 'UniBiFAS_uni_IT':
        model = UniBiFAS_uni_IT_Model(cfg)
    elif mode == 'UniBiFAS_uni_TI':
        model = UniBiFAS_uni_TI_Model(cfg)
    else:
        raise ValueError(f"Unknown model mode: {mode}. Available modes: 'UniBiFAS', 'UniBiFAS_TI', 'UniBiFAS_bi_IT', 'UniBiFAS_bi_TI', 'UniBiFAS_uni_IT', 'UniBiFAS_uni_TI'")
    
    return model


