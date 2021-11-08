# configaformers
A python library for highly configurable transformers - easing model architecture search and experimentation.

Special thanks to lucidrains (https://github.com/lucidrains) and Kharr.

## Notable Features
The main purpose of this library is to allow users to quickly construct transformers by editing config files. We will also provide prebuilt configurations to common or promising model architectures.

Another feature is our model compiler. When a model is initialized it will print out (on your console) all modules, shapes, input and output names. It also performs shape checking which helps catch errors prior to running data through the model.

## Setup
Requirements: PyTorch and einops
```bash
git clone https://github.com/antofuller/configaformers.git
cd /content/configaformers
```

## Usage
Quick demo that will configure a 768-wide, 12-layer transformer, with a language modeling head.

Import, and create token embedding block:

```python
from model_builder import ConfigaFormer
from prebuilt_blocks import get_transformer_block

model_dim = 768
num_heads = 12

# Token embedding block
emb = [{'type': 'embedding',
        'output_dim': model_dim,
        'num_classes': 50257}]
```

Use our prebuilt transformer block:

```python
transformer_block = get_transformer_block(num_heads=num_heads, dim=model_dim)
```

Create language modeling head:

```python
to_logits = [{'type': 'linear',
              'output_dim': 50257,
              'output_name': 'logits'}]
```

Create blocks, initialize input shapes, and init the model:

```python
blocks = [{"config": emb,
           "repeat": 1},
          {"config": transformer_block,
           "repeat": 12},
          {"config": to_logits,
           "repeat": 1},
          ]

my_config = {'blocks': blocks}
input_streams = {'emb_ids': ['B', 'L_in'], 'attn_offset': ['B', num_heads, 'L_in', 'L_in'],}

model = ConfigaFormer(model_config=my_config, input_shapes=input_streams).cuda()
```

This will print out the transformer config:

```bash
Block #1, 1x
embedding -> Input(s): emb_ids (BSZ, L_in) - Output(s): x (BSZ, L_in, 768)


Block #2, 12x
make_stream -> Input(s): x (BSZ, L_in, 768) - Output(s): residual (BSZ, L_in, 768)
norm -> Input(s): x (BSZ, L_in, 768) - Output(s): x (BSZ, L_in, 768)
linear -> Input(s): x (BSZ, L_in, 768) - Output(s): queries (BSZ, L_in, 768)
linear -> Input(s): x (BSZ, L_in, 768) - Output(s): keys (BSZ, L_in, 768)
linear -> Input(s): x (BSZ, L_in, 768) - Output(s): values (BSZ, L_in, 768)
make_heads -> Input(s): queries (BSZ, L_in, 768) - Output(s): queries (BSZ, 12, L_in, 64)
make_heads -> Input(s): keys (BSZ, L_in, 768) - Output(s): keys (BSZ, 12, L_in, 64)
make_heads -> Input(s): values (BSZ, L_in, 768) - Output(s): values (BSZ, 12, L_in, 64)
mha_dots -> Input(s): queries (BSZ, 12, L_in, 64), keys (BSZ, 12, L_in, 64) - Output(s): attn_dots (BSZ, 12, L_in, L_in)
attention_offset -> Input(s): attn_dots (BSZ, 12, L_in, L_in), attn_offset (BSZ, 12, L_in, L_in) - Output(s): attn_dots (BSZ, 12, L_in, L_in)
mha_sum -> Input(s): values (BSZ, 12, L_in, 64), attn_dots (BSZ, 12, L_in, L_in) - Output(s): x (BSZ, 12, L_in, 64)
merge_heads -> Input(s): x (BSZ, 12, L_in, 64) - Output(s): x (BSZ, L_in, 768)
linear -> Input(s): x (BSZ, L_in, 768) - Output(s): x (BSZ, L_in, 768)
merge_streams -> Input(s): x (BSZ, L_in, 768), residual (BSZ, L_in, 768) - Output(s): x (BSZ, L_in, 768)
make_stream -> Input(s): x (BSZ, L_in, 768) - Output(s): residual (BSZ, L_in, 768)
norm -> Input(s): x (BSZ, L_in, 768) - Output(s): x (BSZ, L_in, 768)
linear -> Input(s): x (BSZ, L_in, 768) - Output(s): x (BSZ, L_in, 3072)
activation -> Input(s): x (BSZ, L_in, 3072) - Output(s): x (BSZ, L_in, 3072)
linear -> Input(s): x (BSZ, L_in, 3072) - Output(s): x (BSZ, L_in, 768)
merge_streams -> Input(s): x (BSZ, L_in, 768), residual (BSZ, L_in, 768) - Output(s): x (BSZ, L_in, 768)


Block #3, 1x
linear -> Input(s): x (BSZ, L_in, 768) - Output(s): logits (BSZ, L_in, 50257)
```

Before running, we need to get the attention offset (in this case, AliBi with a causal mask):

```python
from attention_offset_module import get_alibi

attn_offset = get_alibi(num_heads=12)
```

Now we can use the model:

```python
input_data = {'emb_ids': batch_ids.view(bsz, 1024).cuda(),
              'attn_offset': attn_offset.cuda()}

logits = model(input_data)['logits'].view(bsz, 1024, 50257)
```

## Features on the way...
1. Create more prebuilt blocks
2. Improve attention offset helpers
3. Efficient attention (Routing Transformer)
4. Add more experiments
5. Triton for speed-up