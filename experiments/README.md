# Experimenting with *configaformer*
We will start with training a language model (LM) with a sequence length of 1024 on 500 million tokens. The dataset was created by taking text articles from The Pile until the 500 million mark was reached. Articles with less than 1025 tokens were not included - so the model should benefit by longer contexts.

Loss values for every single token, for every config, will be released to support a thorough analysis of how various model architectures learn. 

The two figures below are from the baseline/default model architecture with dim_model=768, and 12 transformer layers (128M non-embedding params).  

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/baseline_position.PNG">

Figure 1

Above, we plot context length (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. 

There are 2 observations that stick out. First, notice that early on in training, loss values are lower for context lengths in the 100-200 range, and slowly increase with increasing context. Later on in training, tokens with longer contexts have lower loss - which is the expected result given that they have more information to make predictions. So maybe early on in training, more information actually confuses the model and degrades performance. The second observation is the strange behaviour in the first 20 tokens, or so. This will be investigated later, but my initial guess is that this is either an artifact of the training data (for example, if 3rd token is typically a sub-word/suffix, then it will be much easier to predict than the start of a new word), and/or is the result of our position encoding strategy.

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/baseline_vocab.PNG">

Figure 2

Above, we plot vocabulary bucket (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. The vocabulary buckets (27 in total) were created by sorting each token by occurrence, in our dataset, and making a new bucket every 20 million tokens (cumulatively). As a result, the first 5 buckets only contain a single token. 

No surprise here, more common tokens are easier to predict by the model. We can also see that more common tokens are learned first, and most of the improvements towards the end of training occur in our least common tokens. 

## Token Shifting w/ RoPE

This section will explore various token shifting configurations while using [rotary position embeddings (RoPE)](https://arxiv.org/abs/2104.09864). The number of features rotated in the attention mechanism will be 1/4 the attention head dimension, unless stated otherwise. In our case this will be 64/4 = 16 features to rotate. The effect of RoPE on attention scores is that it increases the similarity of queries and keys from near-by tokens (along the sequence dimension), and decreases the similarity of tokens that are far apart; thus infusing positional information in the attention map. Of course, without any positional encoding the attention mechanism has no position information, and it would simply operate on a bag (or set) of tokens.

Token shifting essentially slices up each hidden state (along the feature dimension) in the sequence, and swaps out some number of features with neighboring hidden states. It may be best understood by visualizing the operation. Here is a drawing brought to you by MS Paint ;)

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/token_shifting.png">

Above, we have 3 tokens ('soccer', 'is', and 'better') which are each converted into an embedding of size 100 (these are the features representing the corresponding token). The token shifting operation is then performed, which first slices the embeddings, then shifts them over 1 position. Using our notation, this would look like shift=[40, 60]. So, of the 100 features, 60 are kept, and 40 are replaced by 40 features from the previous position (if there is no previous token, the features will be padded with zeros). This token shifting operation is essentially a convolution, and has been used in computer vision publications. For NLP, this simple technique hasn't been published but EleutherAI's discord chat recommends it. For the general case, our token shift notation is:

num_features(t) = [num_features(t-n), ... , num_features(t-1), num_features(t)]



<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/768_shifting.PNG">

Figure 3

It seems that token shifting converges faster for all settings. Next, we can see (while squinting) that blue is no better than black (baseline) at 500M tokens. Comparing blue to orange we can conclude that blue likely didn't keep enough features from t (256 vs 384 for orange). Lastly, red is the clear winner - simply shifting half of the representation from the previous position in the sequence. We can safely conclude that token shifting helps, but too much shifting hurts, or at least offsets the advantages. These findings align with rumors from EleutherAI's discord. 

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/768_wider.PNG">

Figure 4

Here, we plot 2 wider and shallower models, still with 128M non-embedding parameters, against our baseline. Green doesn't perform as well, no surprise here as this high a width/depth ratio is never used. On the other hand, purple's token shifting does make up for its lack of depth. At 500M tokens, purple matches black, and as a result blue (from Fig. 3).

Below are the loss ratios (averaged over the last 10M tokens) of tokens that start a new word divided by tokens that don't (i.e. sub-words or suffixes).


| Colour  | Loss Ratio | Shift Config |
| ------------- | ------------- | ------------- |
| Black  | <b>1.3288</b>  | None
| Red  | 1.3363  | [384, 384]
| Blue  | 1.3310  | [256, 256, 256]
| Orange  | 1.3404  | [128, 256, 384]
| Green  | <b>1.3297</b>  | None
| Purple  | 1.3336  | [128, 384, 512]

The two lowest loss ratios are for the two models that did not use token shifting, <b>implying that token shifting improves sub-word accuracy more than it improves the accuracy of tokens that start a new word.</b> This intuitively makes sense because token shifting allows the model to incorporate information from near-by tokens via a mechanism that is completely separate from attention. And near-by information is presumably more important for sub-word tokens than for tokens that start off a new word.

## Token Shifting w/ AliBi

Similar to the previous section, we will investigate various token shifting configurations while using the [AliBi positional encoding strategy](https://arxiv.org/abs/2108.12409). AliBi is a new position encoding technique that is very simple, intuitive, and is reported to perform on par with RoPE. We use AliBi's default parameters unless stated otherwise.

<img src="https://github.com/muddyrains/muddy-nets/blob/main/experiments/images/768_shifting_alibi_v2.PNG">

This is our first surprising result - not that our baseline RoPE and baseline AliBi finish with the same loss (black and orange), but that <b>it appears that token shifting interacts with Alibi different from RoPE.</b> In this plot, we can see that shift=[128, 256, 384] is equal to shift=[384, 384] throughout training (red is plotted underneath blue). But with RoPE, shift=[128, 256, 384] is clearly inferior to shift=[384, 384]. This finding will need to be investigated. 

## Etc...