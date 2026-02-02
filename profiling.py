from xml.parsers.expat import model
import torch
import logging
import sys
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    logging.info("Using CUDA")
else:
    device = torch.device('cpu')
    logging.info("Using CPU")

# Import UniBiFAS model
from modeling import BuildModel

def create_unibifas_config():
    """
    Create UniBiFAS configuration based on train.py setup
    """
    # Load the default config
    with open('configs/unibifas.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Add required fields that are normally set by command-line args
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['config'] = 'configs/unibifas.yaml'
    cfg['ckpt'] = None
    cfg['op_dir'] = 'output'
    
    return cfg

def test_flops_with_thop():
    """
    Test FLOPs using thop library
    Install with: pip install thop
    """
    try:
        from thop import profile, clever_format
        
        # Create UniBiFAS model
        cfg = create_unibifas_config()
        net = BuildModel(cfg).to(device)
        
        # IMPORTANT: Cache prompts BEFORE profiling to enable fast inference mode
        # This must be done before calling profile() so thop can measure the actual inference path
        net.eval()
        with torch.no_grad():
            net.enable_inference_mode()   # This caches prompts and sets inference_mode=True
        
        # Define input size (batch_size, channels, height, width)
        # UniBiFAS uses 224x224 for CLIP ViT-B/16
        input_size = (1, 3, 224, 224)
        input_tensor = torch.randn(input_size).to(device)
        
        # Calculate FLOPs and parameters
        logging.info("="*60)
        logging.info("UniBiFAS Model (Unified Bidirectional Face Anti-Spoofing)")
        logging.info("="*60)
        logging.info(f"Input size: {input_size}")
        logging.info(f"Backbone: {cfg['model']['backbone']}")
        logging.info(f"Context tokens: {cfg['model']['n_ctx']}")
        logging.info(f"Prompt depth: {cfg['model']['prompt_depth']}")
        logging.info(f"Interaction layers: {cfg['model']['interaction_layers']}")
        logging.info(f"Inference mode: {net.prompt_learner.inference_mode}")
        
        flops, params = profile(net, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.6f")
        
        logging.info(f"Total FLOPs: {flops_str}")
        logging.info(f"Total Parameters: {params_str}")
        
        return True
    except ImportError:
        logging.warning("thop library not found. Install with: pip install thop")
        return False
    except Exception as e:
        logging.error(f"Error in test_flops_with_thop: {e}")
        import traceback
        traceback.print_exc()
        return False

def count_parameters(model, model_name="Model"):
    """
    Count the number of trainable parameters in the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count buffers separately (non-trainable tensors like positional embeddings)
    total_buffers = sum(b.numel() for b in model.buffers())
    
    # Count only parameters (excluding buffers) - this matches thop's count
    params_only = total_params
    
    logging.info(f"{model_name} - Total parameters: {total_params:,} ({total_params / 1e6:.3f} M)")
    logging.info(f"{model_name} - Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
    logging.info(f"{model_name} - Registered buffers: {total_buffers:,} ({total_buffers / 1e6:.3f} M)")
    logging.info(f"{model_name} - Parameters + Buffers: {(total_params + total_buffers):,} ({(total_params + total_buffers) / 1e6:.3f} M)")
    
    return total_params, trainable_params

def measure_inference_time():
    """
    Measure the inference time of the model
    """
    cfg = create_unibifas_config()
    net = BuildModel(cfg).to(device)
    net.eval()
    with torch.no_grad():
        net.enable_inference_mode() 
    
    # Define input (224x224 for CLIP ViT-B/16)
    input_size = (1, 3, 224, 224)
    input_tensor = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = net(input_tensor)
    
    # Measure inference time
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        
        num_runs = 100
        starter.record()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = net(input_tensor)
        ender.record()
        torch.cuda.synchronize()
        
        avg_time = starter.elapsed_time(ender) / num_runs
        logging.info("="*60)
        logging.info("Inference Time (GPU)")
        logging.info("="*60)
        logging.info(f"Average inference time: {avg_time:.3f} ms")
        logging.info(f"FPS: {1000 / avg_time:.2f}")
    else:
        import time
        num_runs = 100
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = net(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        logging.info("="*60)
        logging.info("Inference Time (CPU)")
        logging.info("="*60)
        logging.info(f"Average inference time: {avg_time:.3f} ms")
        logging.info(f"FPS: {1000 / avg_time:.2f}")

if __name__ == "__main__":
    logging.info("="*60)
    logging.info("Testing FLOPs and Parameters for UniBiFAS")
    logging.info("Architecture: UniBiFAS (Unified Bidirectional FAS)")
    logging.info("="*60)
    
    # Count parameters
    logging.info("\n--- Parameter Count ---")
    try:
        cfg = create_unibifas_config()
        net = BuildModel(cfg).to(device)
        
        logging.info("="*60)
        total, trainable = count_parameters(net, "UniBiFAS Model")
        logging.info("="*60)
    except Exception as e:
        logging.error(f"Error counting parameters: {e}")
        import traceback
        traceback.print_exc()
    
    # FLOPs calculation with thop
    logging.info("\n--- FLOPs Calculation (thop) ---")
    success = test_flops_with_thop()
    
    if not success:
        logging.warning("\nFLOPs calculation library not found.")
        logging.warning("Please install: pip install thop")
    
    # Measure inference time
    logging.info("\n--- Inference Time ---")
    try:
        measure_inference_time()
    except Exception as e:
        logging.error(f"Error measuring inference time: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("\n" + "="*60)
    logging.info("Testing completed!")
    logging.info("="*60)
