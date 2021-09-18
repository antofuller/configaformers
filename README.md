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
          'layers': layer_list,
          }
```
- *dim_model* : The main dimension of the model. Each block both receives, and outputs, dim_model number of features per token. In this example, we use a dim_model of 768, but 512, 1024, 1536, and 2048 are also common dim_model values.
- *vocab_size* : The number of tokens in your vocabulary. This is only used when creating the input embedding table, not at the output. In this case, we have 50257 tokens in our vocab - the same as the RoBERTa tokenizer.
- *dim_rope* : The number of features, per token, you wish to rotate. This infuses position information into the attention mechanism. This will be the same for every attention layer, and is typically 1/8 to 1/2 the size of the attention head dimension.
- *classifiers* : A list of integers, where each integer is the number of classes. For language modeling the number of classes should be equal to the size of your vocabulary. This example also includes a classifier with 6 classes that is independent of the language modeling classifier. 
- *input_emb_size* : The number of features in the token embedding table, typically it is equal to dim_model, but if it's not, then a linear projection will be applied to reach dim_model. The total number of embedding parameters is equal to input_emb_size times vocab_size.
- *layers* : A list that configures each transformer layer. See below.

## Blocks to Add:
1. gMLP
2. Alibi position bias
3. Gating attention output

## Features To Add:
1. Add basic training script
2. Incorporate MS DeepSpeed and/or TPU support
3. Add automatic progressive training capability
4. Add dataset management tools
