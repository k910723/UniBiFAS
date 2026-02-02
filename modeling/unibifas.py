import torch
import torch.nn as nn
from clip import clip
from clip.model import QuickGELU
from collections import OrderedDict
import os
import copy
from .prompt_templates import *

# =================================================================================
# Helper Function to Load CLIP
# =================================================================================

def LoadClip(cfg, zero_shot_model=False):
    """
    Loads the CLIP model with custom modifications for prompt learning.
    """
    backbone_name = cfg['model']['backbone']
    model_path = clip._download(clip._MODELS[backbone_name], os.path.expanduser("~/.cache/clip"))

    try:
        model = torch.jit.load(model_path, map_location=cfg.get('device', 'cpu')).eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location=cfg.get('device', 'cpu'))

    # Build a modified CLIP model that accepts prompts in its forward pass
    design_details = {
        "trainer": 'UniBiFAS', # This key enables prompt inputs in the modified CLIP code
        "vision_depth": cfg['model']['prompt_depth'],
        "language_depth": cfg['model']['prompt_depth'],
        "vision_ctx": cfg['model']['n_ctx'],
        "language_ctx": cfg['model']['n_ctx']
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

# =================================================================================
# Core Attention and Prompting Modules
# =================================================================================

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class FeaturePreprocessMLP(nn.Module):
    """MLP to post-process features after LKP modules."""
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
    
    def forward(self, x):
        return x + self.mlp(x)  # Residual connection

class ProjectionMLP(nn.Module):
    """Enhanced projection with MLP."""
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim) * 2
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, cross_prompts_text_deeper):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, cross_prompts_text_deeper]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class AttentionPooling(nn.Module):
    """Layer-specific Knowledge Proxy (LKP)."""
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        token_query = token_query + self.attn(self.ln_1(token_query), self.ln_1(sequence_key), self.ln_1(sequence_value), need_weights=False)[0]
        token_query = self.ln_2(token_query)
        return token_query

class CrossPromptAttention(nn.Module):
    """Multi-scale Knowledge Mapper."""
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size)
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            QuickGELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        q_proj = q_proj + self.attn(self.ln_1(q_proj), self.ln_1(k_proj), self.ln_1(v_proj), need_weights=False)[0]
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj

class HierarchicalPromptLearner(nn.Module):
    """
    Manages and updates the hierarchical prompts for both vision and text encoders.
    Supports multiple classification tasks: binary, attack type, and artifact type.
    """
    def __init__(self, cfg, clip_model):
        super().__init__()
        classnames = cfg['dataset']['classnames']
        attack_types = cfg['dataset'].get('attack_types', ['print attack', 'replay attack', 'mask attack'])
        artifact_types = cfg['dataset'].get('artifact_types', ['color distortion', 'texture artifact', 'geometric artifact'])
        
        self.n_cls_binary = len(classnames)
        self.n_cls_attack = len(attack_types)
        self.n_cls_artifact = len(artifact_types)
        self.n_ctx = cfg['model']['n_ctx']
        self.prompt_depth = cfg['model']['prompt_depth']
        
        # Define layer boundaries for hierarchical interaction
        self.low_level_layers = cfg['model']['interaction_layers'][0]
        self.mid_level_layers = cfg['model']['interaction_layers'][1]
        
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        vis_dim = 768
        dtype = clip_model.dtype
        
        # Initialize prompts and attention mechanisms
        self._initialize_prompts(ctx_dim, vis_dim, dtype)
        self._initialize_attention_mechanisms(ctx_dim, vis_dim)
        self._initialize_knowledge_proxies(ctx_dim, vis_dim, dtype)
        self._initialize_postprocess_mlps(ctx_dim, vis_dim)
        
        # Initialize text prompts with ensembling
        self._initialize_text_prompts(cfg, classnames, attack_types, artifact_types, clip_model, dtype)

        self.inference_mode = False

    def _initialize_prompts(self, ctx_dim, vis_dim, dtype):
        """Initialize text and visual prompts."""
        ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        # Create prompts for each layer
        self.cross_prompts_text = nn.ParameterList(
            [self.ctx] + [nn.Parameter(torch.empty(self.n_ctx, ctx_dim, dtype=dtype)) for _ in range(self.prompt_depth - 1)]
        )
        for p in self.cross_prompts_text[1:]:
            nn.init.normal_(p, std=0.02)
        
        self.cross_prompts_visual = nn.ParameterList(
            [nn.Parameter(torch.empty(self.n_ctx, vis_dim, dtype=dtype)) for _ in range(self.prompt_depth)]
        )
        for p in self.cross_prompts_visual:
            nn.init.normal_(p, std=0.02)

    def _initialize_attention_mechanisms(self, ctx_dim, vis_dim):
        """Initialize attention mechanisms for cross-modal interaction."""
        # Vision -> Text (Low Level)
        self.v2t_low = CrossPromptAttention(ctx_dim, vis_dim, num_attention_heads=8)
        # Vision <-> Text (Mid Level)
        self.v2t_mid = CrossPromptAttention(ctx_dim, vis_dim, num_attention_heads=8)
        self.t2v_mid = CrossPromptAttention(vis_dim, ctx_dim, num_attention_heads=8)
        # Text -> Vision (High Level)
        self.t2v_high = CrossPromptAttention(vis_dim, ctx_dim, num_attention_heads=8)

    def _initialize_knowledge_proxies(self, ctx_dim, vis_dim, dtype):
        """Initialize Layer-specific Knowledge Proxies (LKP)."""
        self.lkp_v_low = _get_clones(AttentionPooling(vis_dim, 8), self.low_level_layers)
        self.lkp_v_mid = _get_clones(AttentionPooling(vis_dim, 8), self.mid_level_layers - self.low_level_layers)
        self.lkp_t_mid = _get_clones(AttentionPooling(ctx_dim, 8), self.mid_level_layers - self.low_level_layers)
        self.lkp_t_high = _get_clones(AttentionPooling(ctx_dim, 8), self.prompt_depth - self.mid_level_layers)

        # Proxy tokens to act as queries in LKP
        self.v_proxy_low = nn.ParameterList([nn.Parameter(torch.randn(1, vis_dim, dtype=dtype)) for _ in range(self.low_level_layers)])
        self.v_proxy_mid = nn.ParameterList([nn.Parameter(torch.randn(1, vis_dim, dtype=dtype)) for _ in range(self.mid_level_layers - self.low_level_layers)])
        self.t_proxy_mid = nn.ParameterList([nn.Parameter(torch.randn(1, ctx_dim, dtype=dtype)) for _ in range(self.mid_level_layers - self.low_level_layers)])
        self.t_proxy_high = nn.ParameterList([nn.Parameter(torch.randn(1, ctx_dim, dtype=dtype)) for _ in range(self.prompt_depth - self.mid_level_layers)])

    def _initialize_postprocess_mlps(self, ctx_dim, vis_dim):
        """Initialize MLP post-processing modules for LKP."""
        self.visual_postprocess_mlp_low = nn.ModuleList([
            FeaturePreprocessMLP(vis_dim) for _ in range(self.low_level_layers)
        ])
        self.visual_postprocess_mlp_mid = nn.ModuleList([
            FeaturePreprocessMLP(vis_dim) for _ in range(self.mid_level_layers - self.low_level_layers)
        ])
        self.text_postprocess_mlp_mid = nn.ModuleList([
            FeaturePreprocessMLP(ctx_dim) for _ in range(self.mid_level_layers - self.low_level_layers)
        ])
        self.text_postprocess_mlp_high = nn.ModuleList([
            FeaturePreprocessMLP(ctx_dim) for _ in range(self.prompt_depth - self.mid_level_layers)
        ])

    def _create_template_ensemble(self, templates, prompt_prefix, clip_model, dtype):
        """Create ensemble embedding from multiple templates."""
        template_embeddings = []
        with torch.no_grad():
            for template in templates:
                prompt = f"{prompt_prefix} {template}."
                tokenized = clip.tokenize(prompt)
                embedding = clip_model.token_embedding(tokenized).type(dtype)
                template_embeddings.append(embedding)
            
            # Average all template embeddings
            ensemble_embedding = torch.mean(torch.stack(template_embeddings), dim=0)
        return ensemble_embedding

    def _replace_prompt_embeddings(self, original_embedding, class_mappings, ensemble_mappings):
        """Replace prompt embeddings based on class and ensemble mappings."""
        embedding = original_embedding.clone()
        
        for i, class_name in enumerate(class_mappings):
            for key, ensemble_emb in ensemble_mappings.items():
                if key.lower() in class_name.lower():
                    # Replace suffix tokens with ensemble embeddings
                    embedding[i, 1+self.n_ctx:] = ensemble_emb[0, 1+self.n_ctx:]
                    break
        
        return embedding

    def _initialize_text_prompts(self, cfg, classnames, attack_types, artifact_types, clip_model, dtype):
        """Initialize text prompts with template ensembling for multiple classification tasks."""
        # --- Tokenization for Text Encoder (Multiple Classification Tasks) ---
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        
        # === Binary Classification ===
        prompts_binary = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized_prompts_binary = torch.cat([clip.tokenize(p) for p in prompts_binary])

        # Create template ensembles for binary classification
        binary_real_ensemble = self._create_template_ensemble(
            FLIP_real_templates, prompt_prefix, clip_model, dtype)
        binary_spoof_ensemble = self._create_template_ensemble(
            FLIP_spoof_templates, prompt_prefix, clip_model, dtype)

        # Create binary embeddings with ensemble replacement
        with torch.no_grad():
            embedding_binary_original = clip_model.token_embedding(tokenized_prompts_binary).type(dtype)
        
        binary_ensemble_mappings = {
            'real': binary_real_ensemble,
            'spoof': binary_spoof_ensemble
        }
        embedding_binary = self._replace_prompt_embeddings(
            embedding_binary_original, classnames, binary_ensemble_mappings)

        # === Attack Type Classification ===
        prompts_attack = [f"{prompt_prefix} {name}." for name in attack_types]
        tokenized_prompts_attack = torch.cat([clip.tokenize(p) for p in prompts_attack])
        
        # Create template ensembles for attack classification
        real_ensemble = self._create_template_ensemble(
            TeG_DG_real_templates, prompt_prefix, clip_model, dtype)
        print_ensemble = self._create_template_ensemble(
            TeG_DG_print_templates, prompt_prefix, clip_model, dtype)
        replay_ensemble = self._create_template_ensemble(
            TeG_DG_replay_templates, prompt_prefix, clip_model, dtype)

        # Create attack embeddings with ensemble replacement
        with torch.no_grad():
            embedding_attack_original = clip_model.token_embedding(tokenized_prompts_attack).type(dtype)
        
        attack_ensemble_mappings = {
            'real': real_ensemble,
            'print': print_ensemble,
            'replay': replay_ensemble
        }
        embedding_attack = self._replace_prompt_embeddings(
            embedding_attack_original, attack_types, attack_ensemble_mappings)

        # === Artifact Type Classification ===
        prompts_artifact = [f"{prompt_prefix} {name}." for name in artifact_types]
        tokenized_prompts_artifact = torch.cat([clip.tokenize(p) for p in prompts_artifact])
        
        with torch.no_grad():
            embedding_artifact = clip_model.token_embedding(tokenized_prompts_artifact).type(dtype)

        # === Register Buffers ===
        # Binary classification
        self.register_buffer('token_prefix_binary', embedding_binary[:, :1, :])
        self.register_buffer('token_suffix_binary', embedding_binary[:, 1 + self.n_ctx:, :])
        self.tokenized_prompts_binary = tokenized_prompts_binary
        
        # Attack classification
        self.register_buffer('token_prefix_attack', embedding_attack[:, :1, :])
        self.register_buffer('token_suffix_attack', embedding_attack[:, 1 + self.n_ctx:, :])
        self.tokenized_prompts_attack = tokenized_prompts_attack
        
        # Artifact classification
        self.register_buffer('token_prefix_artifact', embedding_artifact[:, :1, :])
        self.register_buffer('token_suffix_artifact', embedding_artifact[:, 1 + self.n_ctx:, :])
        self.tokenized_prompts_artifact = tokenized_prompts_artifact

    def construct_prompts(self, ctx, prefix, suffix, n_cls):
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        # --- Hierarchical Cross-Modal Interaction ---

        if not self.inference_mode:
            # 1. Low Level: Vision -> Text (e.g., layers 0-3)
            proxy_v_tokens = []
            for i in range(self.low_level_layers):
                # Apply LKP first, then post-process with MLP
                proxy = self.lkp_v_low[i](self.v_proxy_low[i], self.cross_prompts_visual[i], self.cross_prompts_visual[i])
                proxy = self.visual_postprocess_mlp_low[i](proxy)
                proxy_v_tokens.append(proxy)
            
            proxy_v_low = torch.cat(proxy_v_tokens, dim=0)
            text_prompts_low = torch.cat([p.unsqueeze(0) for p in self.cross_prompts_text[:self.low_level_layers]], dim=0)
            
            updated_text_low = self.v2t_low(text_prompts_low.flatten(0, 1), proxy_v_low, proxy_v_low)
            updated_text_low = updated_text_low.view(self.low_level_layers, self.n_ctx, -1)
            
            for i in range(self.low_level_layers):
                self.cross_prompts_text[i].data.copy_(updated_text_low[i])

            # 2. Mid Level: Vision <-> Text (e.g., layers 4-7)
            mid_range = range(self.low_level_layers, self.mid_level_layers)
            proxy_v_tokens_mid, proxy_t_tokens_mid = [], []
            for i, layer_idx in enumerate(mid_range):
                # Apply LKP first, then post-process with MLPs
                proxy_v = self.lkp_v_mid[i](self.v_proxy_mid[i], self.cross_prompts_visual[layer_idx], self.cross_prompts_visual[layer_idx])
                proxy_v = self.visual_postprocess_mlp_mid[i](proxy_v)
                proxy_v_tokens_mid.append(proxy_v)
                
                proxy_t = self.lkp_t_mid[i](self.t_proxy_mid[i], self.cross_prompts_text[layer_idx], self.cross_prompts_text[layer_idx])
                proxy_t = self.text_postprocess_mlp_mid[i](proxy_t)
                proxy_t_tokens_mid.append(proxy_t)
            
            proxy_v_mid = torch.cat(proxy_v_tokens_mid, dim=0)
            proxy_t_mid = torch.cat(proxy_t_tokens_mid, dim=0)
            
            text_prompts_mid = torch.cat([self.cross_prompts_text[i].unsqueeze(0) for i in mid_range], dim=0)
            visual_prompts_mid = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in mid_range], dim=0)
            
            # V -> T
            updated_text_mid = self.v2t_mid(text_prompts_mid.flatten(0,1), proxy_v_mid, proxy_v_mid).view_as(text_prompts_mid)
            # T -> V
            updated_visual_mid = self.t2v_mid(visual_prompts_mid.flatten(0,1), proxy_t_mid, proxy_t_mid).view_as(visual_prompts_mid)

            for i, layer_idx in enumerate(mid_range):
                self.cross_prompts_text[layer_idx].data.copy_(updated_text_mid[i])
                self.cross_prompts_visual[layer_idx].data.copy_(updated_visual_mid[i])

            # 3. High Level: Text -> Vision (e.g., layers 8-11)
            high_range = range(self.mid_level_layers, self.prompt_depth)
            proxy_t_tokens_high = []
            for i, layer_idx in enumerate(high_range):
                # Apply LKP first, then post-process with MLP
                proxy = self.lkp_t_high[i](self.t_proxy_high[i], self.cross_prompts_text[layer_idx], self.cross_prompts_text[layer_idx])
                proxy = self.text_postprocess_mlp_high[i](proxy)
                proxy_t_tokens_high.append(proxy)
                
            proxy_t_high = torch.cat(proxy_t_tokens_high, dim=0)
            visual_prompts_high = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in high_range], dim=0)
            
            updated_visual_high = self.t2v_high(visual_prompts_high.flatten(0,1), proxy_t_high, proxy_t_high).view_as(visual_prompts_high)

            for i, layer_idx in enumerate(high_range):
                self.cross_prompts_visual[layer_idx].data.copy_(updated_visual_high[i])

        # --- Prepare outputs for CLIP encoders (Multiple Tasks) ---
        # Binary classification prompts
        text_input_prompts_binary = self.construct_prompts(
            self.cross_prompts_text[0], 
            self.token_prefix_binary, 
            self.token_suffix_binary,
            self.n_cls_binary
        )
        
        # Attack classification prompts
        text_input_prompts_attack = self.construct_prompts(
            self.cross_prompts_text[0], 
            self.token_prefix_attack, 
            self.token_suffix_attack,
            self.n_cls_attack
        )
        
        # Artifact classification prompts
        text_input_prompts_artifact = self.construct_prompts(
            self.cross_prompts_text[0], 
            self.token_prefix_artifact, 
            self.token_suffix_artifact,
            self.n_cls_artifact
        )
        
        # Prompts for deeper layers
        deep_text_prompts = [p for p in self.cross_prompts_text[1:]]
        deep_visual_prompts = [p for p in self.cross_prompts_visual[1:]]
        
        return (text_input_prompts_binary, text_input_prompts_attack, text_input_prompts_artifact,
                self.cross_prompts_visual[0], deep_text_prompts, deep_visual_prompts,
                self.tokenized_prompts_binary, self.tokenized_prompts_attack, self.tokenized_prompts_artifact)


# =================================================================================
# Main Model
# =================================================================================

class UniBiFAS_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        clip_model = LoadClip(cfg)
        self.prompt_learner = HierarchicalPromptLearner(cfg, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        # Projection layer for patch tokens to match text feature dimension
        # Vision features: 768, Text features: 512
        self.patch_projection = ProjectionMLP(768, 512)

        self.low_cls_projection = ProjectionMLP(768, 512)
        self.mid_cls_projection = ProjectionMLP(768, 512)
        self.high_cls_projection = ProjectionMLP(768, 512)
        
        # Define interaction layers for patch token extraction
        # Convert from 1-based config indices to 0-based Python indices
        self.interaction_layers = [layer - 1 for layer in cfg['model']['interaction_layers']]  # [3, 7] (0-based)
        self.final_layer = cfg['model']['prompt_depth'] - 1  # 11 for ViT-B/16
        
        # Store intermediate features during forward pass
        self.patch_tokens_cache = {}
        self.cls_tokens_cache = {}
        
        # Register hooks to extract patch tokens from specific layers
        self._register_hooks()
        
        # Freeze CLIP encoders
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad_(False)

        for name, param in self.text_encoder.named_parameters():
            param.requires_grad_(False)
    
    def _register_hooks(self):
        """Register forward hooks to extract patch tokens from specific layers."""
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                
                # --- The Fix ---
                # The output is a list; the feature tensor is the first element.
                feature_tensor = None
                if isinstance(output, list) and output:
                    feature_tensor = output[0]
                elif isinstance(output, torch.Tensor):
                    feature_tensor = output
                # --- End of Fix ---

                # Now, proceed with the extracted tensor
                if feature_tensor is not None and isinstance(feature_tensor, torch.Tensor) and len(feature_tensor.shape) == 3:
                    # Store patch tokens: skip CLS token (1st) and prompt tokens (last 4)
                    # feature_tensor shape: [seq_len, batch_size, hidden_dim] = [201, batch, 768]
                    # We want: [197, batch, 768] -> [1:197] (skip CLS, keep patches, skip prompts)
                    num_prompt_tokens = self.cfg['model']['n_ctx']  # 4
                    end_idx = feature_tensor.shape[0] - num_prompt_tokens  # 201 - 4 = 197
                    patch_only = feature_tensor[1:end_idx, :, :]  # [196, batch, 768] 
                    self.patch_tokens_cache[layer_idx] = patch_only.permute(1, 0, 2)  # [batch, 196, 768]
                    self.cls_tokens_cache[layer_idx] = feature_tensor[0, :, :].unsqueeze(1)  # [batch, 1, 768]
                # IMPORTANT: Always return the original, unmodified output
                return output
            
            return hook
        
        target_layers = self.interaction_layers + [self.final_layer]
        for layer_idx in target_layers:
            if hasattr(self.image_encoder, 'transformer') and hasattr(self.image_encoder.transformer, 'resblocks'):
                if layer_idx < len(self.image_encoder.transformer.resblocks):
                    self.image_encoder.transformer.resblocks[layer_idx].register_forward_hook(make_hook(layer_idx))
            
    def enable_inference_mode(self):   
        self.prompt_learner.inference_mode = True
        # print("Inference mode enabled. Prompts cached, interactions will be skipped.")
    
    def disable_inference_mode(self):
        self.prompt_learner.inference_mode = False
        # print("Inference mode disabled. Dynamic prompt refinement re-enabled.")
    
    def forward(self, image):
        # Clear previous cache
        self.patch_tokens_cache = {}
        self.cls_tokens_cache = {}

        # Get the updated, cross-modally refined prompts for multiple tasks
        (text_prompts_binary, text_prompts_attack, text_prompts_artifact,
         shallow_v_prompt, deep_t_prompts, deep_v_prompts,
         tokenized_binary, tokenized_attack, tokenized_artifact) = self.prompt_learner()

        # Forward pass through CLIP vision encoder (this will trigger our hooks)
        image_features = self.image_encoder(image, shallow_v_prompt, deep_v_prompts)
        
        # Extract patch tokens from all three layers for segmentation
        target_layers = self.interaction_layers + [self.final_layer]  # [3, 7, 11]
        
        patch_tokens_list = []
        for layer_idx in target_layers:
            if layer_idx in self.patch_tokens_cache:
                patch_tokens_list.append(self.patch_tokens_cache[layer_idx])
            else:
                # Fallback: create dummy patch tokens if hook didn't work
                print(f"Warning: Patch tokens not found from layer {layer_idx}. Using zeros.")
                batch_size = image.shape[0]
                num_patches = 196  # 14*14 for ViT-B/16 with 224x224 input
                hidden_dim = 768   # ViT-B/16 hidden dimension
                dummy_patches = torch.zeros(batch_size, num_patches, hidden_dim, 
                                          device=image.device, dtype=image.dtype)
                patch_tokens_list.append(dummy_patches)
        
        # Aggregate patch tokens from all three layers
        patch_tokens = torch.stack(patch_tokens_list, dim=0).mean(dim=0)
        
        # Alternative Method: Weighted sum (uncomment if preferred)
        # weights = torch.tensor([0.3, 0.3, 0.4], device=image.device)  # Give more weight to final layer
        # patch_tokens = sum(w * tokens for w, tokens in zip(weights, patch_tokens_list))

        # Project patch tokens to match text feature dimension (768 -> 512)
        patch_tokens = self.patch_projection(patch_tokens)

        # Get cls tokens from all three layers
        cls_tokens_list = []
        for layer_idx in target_layers:
            if layer_idx in self.cls_tokens_cache:
                cls_tokens_list.append(self.cls_tokens_cache[layer_idx])
            else:
                # Fallback: create dummy cls token if hook didn't work
                print(f"Warning: CLS token not found from layer {layer_idx}. Using zeros.")
                batch_size = image.shape[0]
                hidden_dim = 768
                dummy_cls = torch.zeros(batch_size, 1, hidden_dim, 
                                       device=image.device, dtype=image.dtype)
                cls_tokens_list.append(dummy_cls)

        # Stack cls tokens from all layers
        cls_tokens = torch.cat(cls_tokens_list, dim=1)
        # Project each CLS token separately using the appropriate projection
        cls_tokens_projected = torch.zeros(cls_tokens.shape[0], cls_tokens.shape[1], 512,
                                  device=cls_tokens.device, dtype=cls_tokens.dtype)
        
        # Apply separate projections to each layer's CLS token
        cls_tokens_projected[:,0,:] = self.low_cls_projection(cls_tokens[:,0,:])
        cls_tokens_projected[:,1,:] = self.mid_cls_projection(cls_tokens[:,1,:])
        cls_tokens_projected[:,2,:] = self.high_cls_projection(cls_tokens[:,2,:])
        
        # Normalize each cls token
        cls_tokens = cls_tokens_projected / cls_tokens_projected.norm(dim=-1, keepdim=True)

        # Get text features for different classification tasks
        text_feat_b = self.text_encoder(text_prompts_binary, tokenized_binary, deep_t_prompts)
        text_feat_a = self.text_encoder(text_prompts_attack, tokenized_attack, deep_t_prompts)
        text_feat_art = self.text_encoder(text_prompts_artifact, tokenized_artifact, deep_t_prompts)
        
        # Normalize image features
        img_feat_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Normalize text features
        text_feat_b = text_feat_b / text_feat_b.norm(dim=-1, keepdim=True)
        text_feat_a = text_feat_a / text_feat_a.norm(dim=-1, keepdim=True)
        text_feat_art = text_feat_art / text_feat_art.norm(dim=-1, keepdim=True)
        
        return img_feat_norm, cls_tokens, patch_tokens, text_feat_b, text_feat_a, text_feat_art