# Ablation Study: Guidance Direction

This document describes the ablation study on the direction of cross-modal guidance in UniBiFAS.

## Overview

We investigate cross-modal guidance in UniBiFAS through comprehensive ablation studies examining:
1. **Guidance Direction**: V→T vs T→V (which modality guides which at different depths)
2. **Bidirectional Interaction**: With vs Without (is mid-level bidirectional exchange necessary)
3. **Unidirectional Necessity**: Does T→V guidance contribute at all?

This results in **six model variants**:

### 1. UniBiFAS (Baseline) - V→T / V↔T / T→V [With Bidirectional]
- **Low Level (layers 0-3)**: Vision → Text
  - Visual features guide text prompt learning
- **Mid Level (layers 4-7)**: Vision ↔ Text (bidirectional)
  - Mutual information exchange between modalities
- **High Level (layers 8-11)**: Text → Vision
  - Text semantics guide visual feature refinement

### 2. UniBiFAS_TI - T→V / V↔T / V→T [With Bidirectional, Inverted]
- **Low Level (layers 0-3)**: Text → Vision
  - Text semantics guide early visual features
- **Mid Level (layers 4-7)**: Vision ↔ Text (bidirectional)
  - Mutual information exchange between modalities
- **High Level (layers 8-11)**: Vision → Text
  - Visual features guide late text refinement

### 3. UniBiFAS_bi_IT - V→T / T→V [Without Bidirectional]
- **Low Level (layers 0-6)**: Vision → Text
  - Visual features guide text prompt learning
- **High Level (layers 7-11)**: Text → Vision
  - Text semantics guide visual feature refinement
- **No bidirectional interaction** in middle layers
- **Simplified loss**: Binary + artifact + segmentation (no attack loss)

### 4. UniBiFAS_bi_TI - T→V / V→T [Without Bidirectional, Inverted]
- **Low Level (layers 0-6)**: Text → Vision
  - Text semantics guide early visual features
- **High Level (layers 7-11)**: Vision → Text
  - Visual features guide late text refinement
- **No bidirectional interaction** in middle layers
- **Simplified loss**: Binary + artifact + segmentation (no attack loss)

### 5. UniBiFAS_uni_IT - V→T Only [Unidirectional]
- **All Layers (0-11)**: Vision → Text only
  - Visual features guide text prompts uniformly across all layers
- **No T→V guidance** at any level
- **No bidirectional interaction**
- **Minimal loss**: Binary + segmentation only (no attack/artifact losses)
- **Simplified CLS extraction**: Only final layer (layer 11) CLS token used
  - Unlike other variants that extract multiple CLS tokens (layers 4, 8, 11 for baseline)
  - Tests if single-layer features are sufficient for unidirectional guidance
- Tests if T→V guidance is necessary at all

### 6. UniBiFAS_uni_TI - T→V Only [Unidirectional, Inverted]
- **All Layers (0-11)**: Text → Vision only
  - Text semantics guide visual features uniformly across all layers
- **No V→T guidance** at any level
- **No bidirectional interaction**
- **Minimal loss**: Binary + segmentation only (no attack/artifact losses)
- **Simplified CLS extraction**: Only final layer (layer 11) CLS token used
  - Tests if single-layer features are sufficient for unidirectional guidance
- Tests if V→T guidance is necessary at all

## Implementation

### Model Files
- `modeling/unibifas.py` - Baseline (V→T / V↔T / T→V) with bidirectional
- `modeling/unibifas_TI.py` - Inverted (T→V / V↔T / V→T) with bidirectional
- `modeling/unibifas_bi_IT.py` - Original direction (V→T / T→V) without bidirectional
- `modeling/unibifas_bi_TI.py` - Inverted (T→V / V→T) without bidirectional
- `modeling/unibifas_uni_IT.py` - Unidirectional (V→T only) no T→V
- `modeling/unibifas_uni_TI.py` - Unidirectional inverted (T→V only) no V→T

### Configuration Files
- `configs/unibifas.yaml` - Baseline config (mode: UniBiFAS)
- `configs/unibifas_TI.yaml` - Inverted config (mode: UniBiFAS_TI)
- `configs/unibifas_bi_IT.yaml` - No bidirectional config (mode: UniBiFAS_bi_IT)
- `configs/unibifas_bi_TI.yaml` - Inverted no bidirectional config (mode: UniBiFAS_bi_TI)
- `configs/unibifas_uni_IT.yaml` - Unidirectional config (mode: UniBiFAS_uni_IT)
- `configs/unibifas_uni_TI.yaml` - Unidirectional inverted config (mode: UniBiFAS_uni_TI)

### Model Selection
The model builder automatically selects the correct architecture based on the `mode` parameter in the config:

```yaml
model:
    mode: UniBiFAS        # Baseline: V→T / V↔T / T→V (with bidirectional)
    # or
    mode: UniBiFAS_TI     # Inverted: T→V / V↔T / V→T (with bidirectional)
    # or
    mode: UniBiFAS_bi_IT  # Original direction: V→T / T→V (no bidirectional)
    # or
    mode: UniBiFAS_bi_TI  # Inverted: T→V / V→T (no bidirectional)
    # or
    mode: UniBiFAS_uni_IT # Unidirectional: V→T only (no T→V, no bidirectional)
    # or
    mode: UniBiFAS_uni_TI # Unidirectional inverted: T→V only (no V→T, no bidirectional)
```

## Usage

### Training Baseline Model
```bash
python main.py --config configs/unibifas.yaml --source c --target i
```

### Training Inverted Model (with bidirectional)
```bash
python main.py --config configs/unibifas_TI.yaml --source c --target i
```

### Training No Bidirectional Model (original direction)
```bash
python main.py --config configs/unibifas_bi_IT.yaml --source c --target i
```

### Training No Bidirectional Model (inverted direction)
```bash
python main.py --config configs/unibifas_bi_TI.yaml --source c --target i
```

### Training Unidirectional Model (V→T only)
```bash
python main.py --config configs/unibifas_uni_IT.yaml --source c --target i
```

### Training Unidirectional Model (T→V only, inverted)
```bash
python main.py --config configs/unibifas_uni_TI.yaml --source c --target i
```

### With Randomization and Fixed Seed
```bash
python main.py --config configs/unibifas_TI.yaml --source c --target i --randomize --seed 42
```

## Key Differences in Implementation

### 1. Attention Mechanisms
**Original:**
```python
self.v2t_low = CrossPromptAttention(ctx_dim, vis_dim, ...)  # V→T
self.t2v_high = CrossPromptAttention(vis_dim, ctx_dim, ...)  # T→V
```

**Inverted:**
```python
self.t2v_low = CrossPromptAttention(vis_dim, ctx_dim, ...)  # T→V
self.v2t_high = CrossPromptAttention(ctx_dim, vis_dim, ...)  # V→T
```

**No Bidirectional:**
```python
self.v2t_low = CrossPromptAttention(ctx_dim, vis_dim, ...)  # V→T
self.t2v_high = CrossPromptAttention(vis_dim, ctx_dim, ...)  # T→V
# No mid-level bidirectional mechanisms
```

### 2. Knowledge Proxies
**Original:**
```python
self.lkp_v_low = ...  # Vision proxies for low level
self.lkp_t_high = ...  # Text proxies for high level
```

**Inverted:**
```python
self.lkp_t_low = ...  # Text proxies for low level
self.lkp_v_high = ...  # Vision proxies for high level
```

**No Bidirectional:**
```python
self.lkp_v_low = ...  # Vision proxies for low level (0-6)
self.lkp_t_high = ...  # Text proxies for high level (7-11)
# No mid-level proxies needed
```

### 3. Forward Pass
**Baseline (with bidirectional):**
- Low: Extract visual proxies → Update text prompts
- Mid: Bidirectional exchange
- High: Extract text proxies → Update visual prompts
- CLS tokens: 3 extracted (layers 4, 8, 11)

**Inverted (with bidirectional):**
- Low: Extract text proxies → Update visual prompts
- Mid: Bidirectional exchange
- High: Extract visual proxies → Update text prompts
- CLS tokens: 3 extracted (layers 4, 8, 11)

**No Bidirectional - Original Direction:**
- Low (0-6): Extract visual proxies → Update text prompts
- High (7-11): Extract text proxies → Update visual prompts
- No bidirectional exchange
- CLS tokens: 2 extracted (layers 6, 11 - transition and final)

**No Bidirectional - Inverted:**
- Low (0-6): Extract text proxies → Update visual prompts
- High (7-11): Extract visual proxies → Update text prompts
- No bidirectional exchange
- CLS tokens: 2 extracted (layers 6, 11 - transition and final)

**Unidirectional (V→T only):**
- All (0-11): Extract visual proxies → Update text prompts
- No T→V guidance
- No bidirectional exchange
- CLS tokens: 1 extracted (layer 11 - final only)
- Tests if hierarchical multi-layer features are necessary

## Expected Insights

This comprehensive ablation study helps answer:

### Main Effects:
1. **Guidance Direction Effect (V→T vs T→V):**
   - Does early visual guidance (V→T) outperform early semantic guidance (T→V)?
   - How does guidance direction affect learning of different classification levels?
   
2. **Bidirectional Interaction Effect (With vs Without):**
   - Is the bidirectional middle layer necessary for performance?
   - Can simpler architectures achieve comparable results?

3. **Unidirectional Guidance Necessity:**
   - Is T→V guidance necessary at all? (tested by uni_IT variant)
   - Is V→T guidance necessary at all? (tested by uni_TI variant)
   - Can pure unidirectional guidance achieve competitive performance?

### Interaction Effects:
4. **Combined Impact:**
   - Does the optimal guidance direction depend on whether bidirectional interaction is used?
   - Which combination yields the best performance?

### Task-Specific Effects:
5. **Impact on different classification tasks:**
   - Does guidance direction affect binary, attack-type, and artifact-type classification differently?
   - Which architecture excels at which task?

### Comparison Table:
| Variant | Direction | Bidirectional | T→V | V→T | CLS Tokens | Attack Loss | Artifact Loss | Layers 0-6 | Layer 7 | Layers 8-11 |
|---------|-----------|---------------|-----|-----|------------|-------------|---------------|------------|---------|-------------|
| UniBiFAS | V→T | Yes | Yes | Yes | 3 (4,8,11) | Yes | Yes | V→T/V↔T | V↔T | T→V |
| UniBiFAS_TI | T→V | Yes | Yes | Yes | 3 (4,8,11) | Yes | Yes | T→V/V↔T | V↔T | V→T |
| UniBiFAS_bi_IT | V→T | No | Yes | Yes | 2 (6,11) | **No** | Yes | V→T | Transition | T→V |
| UniBiFAS_bi_TI | T→V | No | Yes | Yes | 2 (6,11) | **No** | Yes | T→V | Transition | V→T |
| UniBiFAS_uni_IT | V→T | No | **No** | Yes | **1 (11)** | **No** | **No** | V→T | V→T | V→T |
| UniBiFAS_uni_TI | T→V | No | Yes | **No** | **1 (11)** | **No** | **No** | T→V | T→V | T→V |

**Note on CLS Token Extraction:**
- The table shows a progressive simplification: 3 tokens → 2 tokens → 1 token
- Baseline variants extract CLS tokens from layers 4, 8, 11 (low/mid/high hierarchy)
- Simplified bi variants extract from layers 6, 11 (transition/final)
- Minimal uni variants (uni_IT, uni_TI) extract only from layer 11 (final layer where guidance has accumulated)
- This tests whether hierarchical multi-level features are necessary for classification

## Results Comparison

Compare the results from both variants across:
- Binary classification (Real vs. Spoof)
- Attack-type classification (Print, Replay, etc.)
- Artifact-type classification (Halftone, Moire, etc.)
- Cross-domain generalization performance
- Convergence speed and stability

## Notes

- Two variants (UniBiFAS, UniBiFAS_TI) use bidirectional interaction in mid-level layers
- Two variants (UniBiFAS_bi_IT, UniBiFAS_bi_TI) remove bidirectional interaction for simplified architecture
- Two variants (UniBiFAS_uni_IT, UniBiFAS_uni_TI) remove opposite guidance entirely to test necessity
  - uni_IT removes T→V guidance to test if V→T alone is sufficient
  - uni_TI removes V→T guidance to test if T→V alone is sufficient
- This creates a comprehensive ablation design testing all key architectural choices
- All other hyperparameters remain identical for fair comparison
- Use the same random seed for reproducible comparisons
- The model builder in `modeling/__init__.py` automatically selects the correct variant based on config
