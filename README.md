# transformers_figuring_out
code to figure out how transormers work

---

## Roadmap (TODO)

### Stage 1: Basic Layers
- [x] Linear (Fully Connected) layer with forward, backward, and parameter updates
- [ ] LayerNorm (layer normalization) with gamma, beta parameters and gradients
- [ ] Activation functions: ReLU, GELU
- [ ] Softmax with backward support

### Stage 2: Attention
- [ ] Scaled Dot-Product Attention (single head)
- [ ] Multi-Head Attention (with splitting and merging heads)

### Stage 3: Transformer Block
- [ ] Feed-Forward Network (FFN) — 2 Linear layers + activation
- [ ] Encoder Block (LayerNorm → MHA → residual → LayerNorm → FFN → residual)

### Stage 4: Full Transformer Encoder
- [ ] Positional Encoding (sinusoidal / learned)
- [ ] Transformer Encoder — stack of multiple Encoder Blocks

### Stage 5: Transformer-based Models
- [ ] Vision Transformer (ViT) — Patch Embedding + Transformer Encoder + classifier head
- [ ] Swin Transformer — windowed attention with shifted windows
- [ ] DETR — Encoder-Decoder + learnable queries + bipartite matching loss

[//]: # (### Stage 6: Training and Infrastructure)

[//]: # (- [ ] Loss functions &#40;Cross-Entropy, MSE&#41;)

[//]: # (- [ ] Optimizers &#40;SGD, Adam&#41;)

[//]: # (- [ ] Training loop &#40;trainer&#41;)

[//]: # (- [ ] Dataset loading and preprocessing &#40;CIFAR-10, MNIST&#41;)

---

## Project Structure

