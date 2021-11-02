# Experimenting with *configaformer*
We will start with training a language model (LM) with a sequence length of 1024 on 500 million tokens. The dataset was created by taking text articles from The Pile until the 500 million mark was reached. Articles with less than 1025 tokens were not included - so the model should benefit by longer contexts.

Loss values for every single token, for every config, will be released to support a thorough analysis of how various model architectures learn. 

The two figures below are from the baseline/default model architecture with dim_model=768, and 12 transformer layers (128M non-embedding params).  

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/baseline_position.PNG">

Figure 1

Above, we plot context length (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. 

There are 2 observations that stick out. First, notice that early on in training, loss values are lower for context lengths in the 100-200 range, and slowly increase with increasing context. Later on in training, tokens with longer contexts have lower loss - which is the expected result given that they have more information to make predictions. So maybe early on in training, more information actually confuses the model and degrades performance. The second observation is the strange behaviour in the first 30 tokens, or so. This will be investigated later, but my initial guess is that this is either an artifact of the training data (for example, if 3rd token is typically a sub-word/suffix, then it will be much easier to predict than the start of a new word), and/or is the result of our position encoding strategy.

Let's do a quick investigation of the role of token type on the strange loss values with contexts less than 30, noted above. 

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/first_30_positions.PNG">

Figure 2

The y-axis plots our baseline loss values at position x, normalized between 0 and 1 (over the full sequence). Red is the normalized ratio of tokens that start a word to tokens that don't. The higher red is, the more tokens at this location start-off a word, and are therefore *naively* more difficult. However, red doesn't correlate with blue/green as much as I hoped. Specifically, positions 2, 3, and 4 have abnormally low loss values - if we only consider our naive token difficulty metric. Secondly, because both RoPE and AliBi position encoding strategies show the same pattern, it doesn't look like this is caused by our choice of position encoding. My next guess is that this phenomenon is caused by the fact that the start of a sequence is also the start of an article in The Pile. So if some articles are started in a similar way, like "CNN News for...", then some early positions may be easier to predict than our naive difficulty metric would suggest.

We should also note that AliBi outperforms RoPE after position 4, but at much longer contexts, RoPE outperforms AliBi (not plotted). This implies that AliBi has greater local bias, and does not benefit from more context as much as RoPE. This result would likely change by adjusting either RoPE, or AliBi's parameters. This is not surprising to me, so it won't be further investigated right now. 


<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/baseline_vocab.PNG">

Figure 3

Above, we plot vocabulary bucket (x-axis) vs loss (y-axis). Across 10 different training points (every 50M tokens). The red-to-blue colour transition reflects the first-to-last batch of 50M tokens. The vocabulary buckets (27 in total) were created by sorting each token by occurrence, in our dataset, and making a new bucket every 20 million tokens (cumulatively). As a result, the first 5 buckets only contain a single token. 

No surprise here, more common tokens are easier to predict by the model. We can also see that more common tokens are learned first, and most of the improvements towards the end of training occur in our least common tokens. 

## Token Shifting w/ RoPE

This section will explore various token shifting configurations while using [rotary position embeddings (RoPE)](https://arxiv.org/abs/2104.09864). The number of features rotated in the attention mechanism will be 1/4 the attention head dimension, unless stated otherwise. In our case this will be 64/4 = 16 features to rotate. The effect of RoPE on attention scores is that it increases the similarity of queries and keys from near-by tokens (along the sequence dimension), and decreases the similarity of tokens that are far apart; thus infusing positional information in the attention map. Of course, without any positional encoding the attention mechanism has no position information, and it would simply operate on a bag (or set) of tokens.

Token shifting essentially slices up each hidden state (along the feature dimension) in the sequence, and swaps out some number of features with neighboring hidden states. It may be best understood by visualizing the operation. Here is a drawing brought to you by MS Paint ;)

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/token_shifting.png">

Above, we have 3 tokens ('soccer', 'is', and 'better') which are each converted into an embedding of size 100 (these are the features representing the corresponding token). The token shifting operation is then performed, which first slices the embeddings, then shifts them over 1 position. Using our notation, this would look like shift=[40, 60]. So, of the 100 features, 60 are kept, and 40 are replaced by 40 features from the previous position (if there is no previous token, the features will be padded with zeros). This token shifting operation is essentially a convolution, and has been used in computer vision publications. For NLP, this simple technique hasn't been published but EleutherAI's discord chat recommends it. For the general case, our token shift notation is:

num_features(t) = [num_features(t-n), ... , num_features(t-1), num_features(t)]



<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/768_shifting.PNG">

Figure 4

It seems that token shifting converges faster for all settings. Next, we can see (while squinting) that blue is no better than black (baseline) at 500M tokens. Comparing blue to orange we can conclude that blue likely didn't keep enough features from t (256 vs 384 for orange). Lastly, red is the clear winner - simply shifting half of the representation from the previous position in the sequence. We can safely conclude that token shifting helps, but too much shifting hurts, or at least offsets the advantages. These findings align with rumors from EleutherAI's discord. 

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/768_wider.PNG">

Figure 5

Here, we plot 2 wider and shallower models, still with 128M non-embedding parameters, against our baseline. Green doesn't perform as well, no surprise here as this high a width/depth ratio is never used. On the other hand, purple's token shifting does make up for its lack of depth. At 500M tokens, purple matches black, and as a result blue (from Fig. 4).

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

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/768_shifting_alibi_v2.PNG">

Figure 6

This is our first surprising result - not that our baseline RoPE and baseline AliBi finish with the same loss (black and orange), but that <b>it appears that token shifting interacts with Alibi different from RoPE.</b> In this plot, we can see that shift=[128, 256, 384] is equal to shift=[384, 384] throughout training (red is plotted underneath blue). But with RoPE, shift=[128, 256, 384] is clearly inferior to shift=[384, 384]. This finding will need to be investigated.

Additionally, performing the token shift operation only on the intermediate FFN representations, shown in green and cyan, clearly hurts performance. So it seems that token shifting inside the FFN limits the FFN's ability to process the attention output (since the FFN is directly after the attention block). 

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/768_shift_add_mult.PNG">

Figure 7

Above, we play around with different ways of performing the token shift operation, and compare them to black and blue (as baselines). The mult/add refers to the operation between the token shifted and non-token shifted representations, and the sigmoid (if applied) is performed only on the token shifted representations. None of them improve on our baseline (black), which implies that performing these more complex interactions between neighboring tokens confuses the model by entangling representations. These experiments were motivated by the success of some gating operations (like gated linear units), although they may work with an explicit GLU - albeit at the cost of extra parameters. Finally, one minor observation here is that performing a sigmoid on the shifted tokens helps when multiplying, but hurts when adding.

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/768_shift_locations.PNG">

Figure 8

Changing the location of the token shift operation seems to significantly alter performance. Up until now, we've always been applying the token shift operation prior to the attention layer (our default setting). Now we will compare this default setting (blue) to four other configurations - all of which are worse than our non-shifted baseline (black). Red is the worst, presumably because it shifts "too much" - shifting at twice the number of spots as blue (24 vs 12 total token shifts). Green's poor performance was surprising because it shifts the exact same amount as blue, it just performs the shift prior to the FFN, rather than the attention layer. I had two potential hypotheses for this. First, the last token shift is closer to the logit classifier for FFN only vs attention only shifting. I thought this was possible because typically the last few layers of a deep network focus on prediction, while the earlier layers tend to build more general representations of the input data. This hypothesis is likely false because when we remove token shift operations from the last (orange) and last two (purple) layers, performance is still noticeably worse than even our non-shifted baseline. The remaining hypothesis is simply that shifting prior to the FFN hinders the FFN's ability to process the attention output - which seems to be a crucial ingredient in transformers. 

Next we will experiment with token shifting at certain layers - but only prior to attention, as FFN token shifting doesn't seem to work well. 

<img src="https://github.com/antofuller/configaformers/blob/main/experiments/language_modeling/images/768_shift_more_locations.PNG">

Figure 9

This plot is surprising to me. It shows that, at the 500M token mark, token shifting every 2nd layer (orange) outperforms token shifting only the first half (red), only the second half (green), and even shifting all layers (blue, which was our previous best model). Considering that two models (green and orange) outperformed blue while only shifting at 6 locations, instead of 12, it seems that blue was over-shifted. Another unexpected result is that shifting the first 6 layers performed worse than shifting the final 6 layers because typically when convolutions and attention blocks are used together, convolutions appear in earlier layers (see the many conv+attn vision models, and DeBERTa-V2 for NLP). Since our model weights are randomly initialized, we should perform several training runs to confirm the result.

| Config  | Mean Loss on tokens 400-500M | Num Runs | Standard Deviation |
| ------------- | ------------- | ------------- | ------------- |
| AliBi, No Shift  | 3.691  | 1 | -
| AliBi, Shift=[384, 384]  | 3.641  | 5 | 0.0036
| AliBi, Shift=[384, 384], every 2nd layer  | <b>3.633</b>  | 5 | 0.0035

This confirms that, at least for configurations tested, skipping the token shift operation every 2nd layer modestly improves performance; it also shows that we can generally rely on single training runs as the standard deviation across 5 runs is small. However, I do expect this result to change if we alter the shift config from [384, 384] to something like [368, 400], which shifts fewer features. I suspect that a 50/50 shift ratio, when shifting only 1 token is "shifting too much" if used at every layer. 

## To be continued...