# Experimenting with *configaformer*
We will start with training a language model (LM) with a sequence length of 1024 on 500 million tokens. The dataset was created by taking text articles from The Pile until the 500 million mark was reached. Articles with less than 1025 tokens were not included - so the model should benefit by longer contexts.

Loss values for every single token, for every config, will be released to support a thorough analysis of how various model architectures learn. 

The two figures below are from the baseline/default model architecture with dim_model=768, and 12 transformer layers (128M non-embedding params).  

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/baseline_position.PNG">

Above, we plot context length (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. 

There are 2 observations that stick out. First, notice that early on in training, loss values are lower for context lengths in the 100-200 range, and slowly increase with increasing context. Later on in training, tokens with longer contexts have lower loss - which is the expected result given that they have more information to make predictions. So maybe early on in training, more information actually confuses the model and degrades performance. The second observation is the strange behaviour in the first 20 tokens, or so. This will be investigated later, but my initial guess is that this is either an artifact of the training data (for example, if 3rd token is typically a sub-word/suffix, then it will be much easier to predict than the start of a new word), and/or is the result of our position encoding strategy.

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/baseline_vocab.PNG">

Above, we plot vocabulary bucket (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. The vocabulary buckets (27 in total) were created by sorting each token by occurrence, in our dataset, and making a new bucket every 20 million tokens (cumulatively). As a result, the first 5 buckets only contain a single token. 

No surprise here, more common tokens are easier to predict by the model. We can also see that more common tokens are learned first, and most of the improvements towards the end of training occur in our least common tokens. 

## Token Shifting

Groups of token shifting configurations:
1. Amount of tokens, number of tokens, skipping tokens, shift location, etc.
2. Model scaling (width, depth)
3. Different configurations at different depths

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/768_shifting.PNG">

Above, we plot training progress in millions of tokens (x-axis) vs loss (y-axis). The shift list is the token shifting strategy -> [num_features @ t-n, ..., num_features @ t-1, num_features @ t]. For example, shift=[128, 256, 384] means 128 features from t-2, and 256 features from t-1 are inserted into t's representation at each layer - finally, 384 features from t are kept. <b>Token shifting is essentially a convolution.</b> 

It seems that token shifting converges faster for all settings. Next, we can see (while squinting) that blue is no better than black (baseline) at 500M tokens. Comparing blue to orange we can conclude that blue likely didn't keep enough features from t (256 vs 356 for orange). Lastly, red is the clear winner. We can safely conclude that token shifting helps, but too much shifting hurts, or at least offsets the advantages. These findings align with rumors from EleutherAI's discord. 

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/768_wider.PNG">

Here, we plot 2 wider and shallower models, still with 128M non-embedding parameters, against our baseline. Green doesn't perform as well, no surprise here as this high a width/depth ratio is never used. On the other hand, purple's token shifting does make up for green's lack of depth. At 500M tokens, purple matches black, and as a result blue (from the previous plot).

Below are the loss ratios (averaged over the last 10M tokens) of tokens that start a new word divided by tokens that don't (i.e. sub-words or suffixes).


| Colour  | Loss Ratio | Shift Config |
| ------------- | ------------- | ------------- |
| Black  | 1.3288  | None
| Red  | 1.3363  | [384, 384]
| Blue  | 1.3310  | [256, 256, 256]
| Orange  | 1.3404  | [128, 256, 384]
| Green  | 1.3297  | None
| Purple  | 1.3336  | [128, 384, 512]

The two lowest loss ratios are for the two models that did not use token shifting, implying that token shifting improves sub-word accuracy more than it improves the accuracy of tokens that start a new word. This intuitively makes sense because token shifting allows the model to incorporate information from near-by tokens via a mechanism that is completely separate from attention.

## Rotary Position Embeddings (RoPE)

Groups of RoPE configurations:
1. Amount of features to rotate, qk vs v rotations, etc.
2. Model scaling (width, depth)
3. Different configurations at different depths

## Etc...