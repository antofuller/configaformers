# Experimenting with *configaformer*
We will start with training a language model (LM) with a sequence length of 1024 on 500 million tokens. The dataset was created by taking text articles from The Pile until the 500 million mark was reached. Articles with less than 1025 tokens were not included - so the model should benefit by longer contexts.

Loss values for every single token, for every config, will be released to support a thorough analysis of how various model architectures learn. 

The two figures below are from the baseline/default model architecture with dim_model=768, and 12 transformer layers (128M non-embedding params).  

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/baseline_position.PNG">

Above, we plot context length (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. 

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/baseline_vocab.PNG">

Above, we plot vocabulary bucket (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. The vocabulary buckets (27 in total) were created by sorting each token by occurrence, in our dataset, and making a new bucket every 20 million tokens (cumulatively). As a result, the first 5 buckets only contain a single token. 

Future plots will be normalized with respect to the baseline plots above, in order to better visualize the differences in model behaviour.

## Token Shifting

Groups of token shifting configurations:
1. Amount of tokens, number of tokens, skipping tokens, shift location, etc.
2. Model scaling (width, depth)
3. Different configurations at different depths

## Rotary Position Embeddings (RoPE)

Groups of RoPE configurations:
1. Amount of features to rotate, qk vs v rotations, etc.
2. Model scaling (width, depth)
3. Different configurations at different depths

## Etc...