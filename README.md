# *configaformer*
A PyTorch library for highly configurable transformers - easing model architecture search and experimentation. Heavily inspired by https://github.com/lucidrains/x-transformers

The implementation prioritizes flexibility and readability over speed.

The aim of this library is not to search for SOTA architectures - if you only care about accuracy use a massive transformer (maybe with RoPE and GEGLU). Rather, this library attempts to facilitate searching for transformer architectures that meet your, task-specific, accuracy-memory-speed requirements. It's also a great way to build intuition about various building blocks by enabling rapid experimentation. 
## Usage
Build your transformer by creating a *config* that contains both entire-model and layer-specific features. Let's start with an example:
```
config = {'dim_model': 768,
          'vocab_size': 50257,
          'dim_rope': 64,
          'classifiers': [50257, 6],
          'input_emb_size': 512,
          'blocks': block_list,
          }
```
- *dim_model* : The main dimension of the model. Each block both receives, and outputs, dim_model number of features per token. In this example, we use a dim_model of 768, but 512, 1024, 1536, and 2048 are also common dim_model values.
- *vocab_size* : The number of tokens in your vocabulary. This is only used when creating the input embedding table, not at the output. In this case, we have 50257 tokens in our vocab - the same as the RoBERTa tokenizer.
- *dim_rope* : The number of features, per token, you wish to rotate. This infuses position information into the attention mechanism. This will be the same for every attention layer, and is typically 1/8 to 1/2 the size of the attention head dimension.
- *classifiers* : A list of integers, where each integer is the number of output classes. For language modeling the number of classes should be equal to the size of your vocabulary. This example also includes a classifier with 6 classes that is independent of the language modeling classifier. The entire classifier block consists of a vanilla FFN followed by a linear mapping to the output classes.
- *input_emb_size* : The number of features in the token embedding table, typically it is equal to dim_model, but if it's not, then a linear projection will be applied to reach dim_model. The total number of embedding parameters is equal to input_emb_size times vocab_size.
- *blocks* : A list that configures each block. See below.
```
block_list = [{"type": "Attention",
               "attn_dim": 1024,
               "num_heads": 16,
               "token_shift_config": shift_input},
              
              {"type": "FFN",
               "ff_mult": 4,
               "inner_token_shift_config": shift_input_inner},
              ]*12
```
Configs for all block types:
- *type* : Either Attention or FFN.
- *pre_norm_bool* : Apply a layer normalization before the block's main operation. Defaults to true.
- *post_norm_bool* : Apply a layer normalization after the block's main operation, but before the skip connection is added. Defaults to false.
- *token_shift_config* : Apply a token shift before the block's main operation. If pre_norm_bool is true, then the token shift will be performed after the layer norm. See below for details.

Configs for attention blocks only:
- *attn_dim* : The dimension for the attention mechanism. It typically is equal to dim_model, and must be evenly divisible by num_heads. The head dimension will be equal to attn_dim divided by num_heads.
- *num_heads* : The number of heads, in the multi-headed attention mechanism. The less num_heads, the greater each head size will be, for a constant attn_dim.
- *previous_attention_bool* : Enables re-using the attention map from the last time it was calculated - ala LazyFormer. Enabling this will skip the entire attention map calculation - it will skip creating queries and keys, and just create values which will be aggregated based on the last attention map available. Defaults to false.
- *residual_attention_bool* : Setting this to true will add a skip connection to the attention map, before the softmax. The added attention map will be taken from the last time an attention map was calculated. The number of attention heads (num_heads) must be equal to num_heads from the last attention block. Otherwise, the model will try to add attention maps of different sizes. No bueno!
- 
## Blocks to Add:
1. Alibi position bias


## Features To Add:
1. Basic training script
2. Incorporate MS DeepSpeed and/or TPU support
3. Automatic progressive training capability
4. Dataset management tools
