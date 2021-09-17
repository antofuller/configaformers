# *configaformer*
A PyTorch library for highly configurable transformers - easing model architecture search and experimentation. Heavily inspired by https://github.com/lucidrains/x-transformers

The implementation prioritizes flexibility and readability over speed.

The aim of this library is not to search for SOTA architectures - for that, use a large vanilla transformer (maybe with RoPE and GEGLU). Rather, this library attempts to facilitate searching for transformer architectures that meet your, task-specific, performance-memory-speed requirements. It's also a great way to build intuition about various building blocks by enabling rapid experimentation. 
## Blocks to Add:
1. gMLP
2. Alibi position bias
3. Gating attention output

## Features To Add:
1. Add basic training script
2. Incorporate MS DeepSpeed and/or TPU support
3. Add automatic progressive training capability
4. Add dataset management tools
