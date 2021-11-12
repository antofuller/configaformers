def ffn_block(dim,
              input_name='x',
              output_name='x',
              ff_mult=4,
              drop_prob=0,
              inner_norm=False,
              init_final_bias=False):

    block = []
    # Project up
    block.append({'type': 'linear', 'input_name': input_name, 'output_name': output_name, 'output_dim': dim * ff_mult})

    # Activation function
    block.append({'type': 'activation', 'input_name': output_name, 'output_name': output_name})

    # Optional dropout
    if drop_prob > 0:
        block.append({'type': 'dropout', 'prob': drop_prob, 'input_name': output_name, 'output_name': output_name})

    # Optional normalize the inner FFN representation (introduced in NormFormer)
    if inner_norm:
        block.append({'type': 'norm', 'input_name': output_name, 'output_name': output_name, 'norm_type': 'layer_norm'})

    # Project back down to dim
    if init_final_bias != False:
        block.append({'type': 'linear', 'input_name': output_name, 'output_name': output_name, 'output_dim': dim, 'init_bias': init_final_bias})
    else:
        block.append({'type': 'linear', 'input_name': output_name, 'output_name': output_name, 'output_dim': dim})

    return block


def attention_block(num_heads,
                    input_name='x',
                    output_name='x',
                    use_attn_offset=True,
                    offset_name='attn_offset',
                    scale_heads=False,
                    mix=True):
    block = []
    # Make QKVs via linear projections
    block.append({'type': 'linear', 'input_name': input_name, 'output_name': 'queries'})
    block.append({'type': 'linear', 'input_name': input_name, 'output_name': 'keys'})
    block.append({'type': 'linear', 'input_name': input_name, 'output_name': 'values'})

    # Rearrange the QKVs into heads
    block.append({'type': 'make_heads', 'input_name': 'queries', 'output_name': 'queries', 'num_heads': num_heads})
    block.append({'type': 'make_heads', 'input_name': 'keys', 'output_name': 'keys', 'num_heads': num_heads})
    block.append({'type': 'make_heads', 'input_name': 'values', 'output_name': 'values', 'num_heads': num_heads})

    # Calculate attention matrix
    block.append({'type': 'mha_dots',
                  'input_name_queries': 'queries',
                  'input_name_keys': 'keys',
                  'output_name': 'attn_dots'})

    if use_attn_offset:  # Apply attention masking and/or biasing (introduced in AliBi)
        block.append({'type': 'attention_offset',
                      'input_name_attn_dots': 'attn_dots',
                      'input_name_attn_offset': offset_name,
                      'output_name': 'attn_dots'})

    # Perform a weighted sum with attention scores and values
    block.append({'type': 'mha_sum',
                  'input_name_values': 'values',
                  'input_name_attn_dots': 'attn_dots',
                  'output_name': output_name})

    if scale_heads:  # Scale each head (introduced in NormFormer)
        block.append({'type': 'scale_dim', 'dim_to_scale': 1, 'input_name': output_name, 'output_name': output_name})

    # Rearrange tensor by merging heads
    block.append({'type': 'merge_heads', 'input_name': output_name, 'output_name': output_name})

    if mix:  # Final linear projection
        block.append({'type': 'linear', 'input_name': output_name, 'output_name': output_name})

    return block


def transformer_block(num_heads,
                      dim,
                      input_name='x',
                      output_name='x',
                      ff_mult=4,
                      use_attn_offset=True,
                      offset_name='attn_offset',
                      drop_prob=0,
                      ):
    block = [] # Make list

    # Make residual and norm
    block.append({'type': 'make_stream', 'input_name': input_name, 'output_name': 'residual'})
    block.append({'type': 'norm', 'input_name': input_name, 'output_name': output_name, 'norm_type': 'layer_norm'})

    # Perform vanilla Multi-Head Attention
    block += attention_block(num_heads=num_heads,
                             input_name=output_name,
                             output_name=output_name,
                             use_attn_offset=use_attn_offset,
                             offset_name=offset_name,
                             scale_heads=False,
                             mix=True)

    # Add residual
    block.append({'type': 'merge_streams',
                  'input_name_1': output_name,
                  'input_name_2': 'residual',
                  'merge_type': 'add',
                  'output_name': output_name})

    # Make residual and norm
    block.append({'type': 'make_stream', 'input_name': output_name, 'output_name': 'residual'})
    block.append({'type': 'norm', 'input_name': output_name, 'output_name': output_name, 'norm_type': 'layer_norm'})

    # FFN
    block += ffn_block(dim=dim,
                       input_name=output_name,
                       output_name=output_name,
                       ff_mult=ff_mult,
                       drop_prob=drop_prob)

    # Add residual
    block.append({'type': 'merge_streams',
                  'input_name_1': output_name,
                  'input_name_2': 'residual',
                  'merge_type': 'add',
                  'output_name': output_name})

    return block


def normformer_block(num_heads,
                     dim,
                     input_name='x',
                     output_name='x',
                     ff_mult=4,
                     use_attn_offset=True,
                     offset_name='attn_offset',
                     drop_prob=0,
                     use_weighted_residual=True,
                     ):
    block = []  # Make list

    # Make residual and norm
    block.append({'type': 'make_stream', 'input_name': input_name, 'output_name': 'residual'})
    block.append({'type': 'norm', 'input_name': input_name, 'output_name': output_name, 'norm_type': 'layer_norm'})

    # Perform HeadScale MHA
    block += attention_block(num_heads=num_heads,
                             input_name=output_name,
                             output_name=output_name,
                             use_attn_offset=use_attn_offset,
                             offset_name=offset_name,
                             scale_heads=True,
                             mix=True)

    # Post attention layer norm
    block.append({'type': 'norm', 'input_name': output_name, 'output_name': output_name, 'norm_type': 'layer_norm'})

    # Add residual
    block.append({'type': 'merge_streams',
                  'input_name_1': output_name,
                  'input_name_2': 'residual',
                  'merge_type': 'add',
                  'output_name': output_name})

    # Make residual and norm
    block.append({'type': 'make_stream', 'input_name': output_name, 'output_name': 'residual'})
    block.append({'type': 'norm', 'input_name': output_name, 'output_name': output_name, 'norm_type': 'layer_norm'})

    # FFN
    block += ffn_block(dim=dim,
                       input_name=output_name,
                       output_name=output_name,
                       ff_mult=ff_mult,
                       drop_prob=drop_prob,
                       inner_norm=True)

    # Optionally scale the residual before adding it
    if use_weighted_residual:
        block.append({'type': 'scale_dim', 'dim_to_scale': 2, 'input_name': 'residual', 'output_name': 'residual'})

    # Add residual
    block.append({'type': 'merge_streams',
                  'input_name_1': output_name,
                  'input_name_2': 'residual',
                  'merge_type': 'add',
                  'output_name': output_name})
    return block


def copygate_block_old(num_heads,
                   dim,
                   main_stream='x',
                   ff_mult_data=4,
                   ff_mult_gate=1,
                   use_attn_offset=True,
                   offset_name='attn_offset',
                   ):
    block = [] # Make list

    # Use naming convention from paper (this is the input to MHA)
    block.append({'type': 'make_stream', 'input_name': main_stream, 'output_name': 'h'})

    # Perform MHA (final output is a)
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': 'queries'})
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': 'keys'})
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': 'values'})

    block.append({'type': 'make_heads', 'input_name': 'queries', 'output_name': 'queries', 'num_heads': num_heads})
    block.append({'type': 'make_heads', 'input_name': 'keys', 'output_name': 'keys', 'num_heads': num_heads})
    block.append({'type': 'make_heads', 'input_name': 'values', 'output_name': 'values', 'num_heads': num_heads})

    block.append({'type': 'mha_dots',
                  'input_name_queries': 'queries',
                  'input_name_keys': 'keys',
                  'output_name': 'attn_dots'})
    if use_attn_offset:  # used for attention mask or AliBi-style biasing
        block.append({'type': 'attention_offset',
                      'input_name_attn_dots': 'attn_dots',
                      'input_name_attn_offset': offset_name,
                      'output_name': 'attn_dots'})
    block.append({'type': 'mha_sum',
                  'input_name_values': 'values',
                  'input_name_attn_dots': 'attn_dots',
                  'output_name': main_stream})

    block.append({'type': 'merge_heads', 'input_name': main_stream, 'output_name': main_stream})
    block.append({'type': 'merge_streams',
                  'input_name_1': main_stream,
                  'input_name_2': 'h',
                  'merge_type': 'add',
                  'output_name': main_stream})
    block.append({'type': 'norm', 'input_name': main_stream, 'output_name': 'a', 'norm_type': 'layer_norm'})

    # FFN data (final output is h_hat)
    block.append({'type': 'linear', 'input_name': 'a', 'output_name': 'h_hat', 'output_dim': dim * ff_mult_data})
    block.append({'type': 'activation', 'input_name': 'h_hat', 'output_name': 'h_hat'})
    block.append({'type': 'linear', 'input_name': 'h_hat', 'output_name': main_stream, 'output_dim': dim})
    block.append({'type': 'norm', 'input_name': 'h_hat', 'output_name': 'h_hat', 'norm_type': 'layer_norm'})

    # FFN gate (final output is g)
    block.append({'type': 'linear', 'input_name': 'a', 'output_name': 'g', 'output_dim': dim * ff_mult_gate})
    block.append({'type': 'activation', 'input_name': 'g', 'output_name': 'g'})
    block.append({'type': 'linear', 'input_name': 'g', 'output_name': 'g', 'output_dim': dim})
    block.append({'type': 'activation', 'activation_function': 'sigmoid', 'input_name': 'g', 'output_name': 'g'})

    block.append({'type': 'merge_streams',
                  'input_name_1': main_stream,
                  'input_name_2': 'residual',
                  'merge_type': 'add',
                  'output_name': main_stream})
    return block


def copygate_block(num_heads,
                   dim,
                   input_name='x',
                   output_name='x',
                   ff_mult_data=4,
                   ff_mult_gate=1,
                   use_attn_offset=True,
                   offset_name='attn_offset',
                   drop_prob=0,
                   ):
    block = [] # Make list

    # Make residual
    block.append({'type': 'make_stream', 'input_name': input_name, 'output_name': 'block_input'})

    # Perform vanilla Multi-Head Attention
    block += attention_block(num_heads=num_heads,
                             input_name=input_name,
                             output_name='a',
                             use_attn_offset=use_attn_offset,
                             offset_name=offset_name,
                             scale_heads=False,
                             mix=True)

    # Add residual
    block.append({'type': 'merge_streams',
                  'input_name_1': 'a',
                  'input_name_2': 'block_input',
                  'merge_type': 'add',
                  'output_name': 'a'})

    # Post residual layer norm
    block.append({'type': 'norm', 'input_name': 'a', 'output_name': 'a', 'norm_type': 'layer_norm'})

    # FFN data with a layer norm after the FFN
    block += ffn_block(dim=dim,
                       input_name='a',
                       output_name='h_hat',
                       ff_mult=ff_mult_data,
                       drop_prob=drop_prob)
    block.append({'type': 'norm', 'input_name': 'h_hat', 'output_name': 'h_hat', 'norm_type': 'layer_norm'})

    # FFN gate with a sigmoid after the FFN
    block += ffn_block(dim=dim,
                       input_name='a',
                       output_name='g',
                       ff_mult=ff_mult_gate,
                       drop_prob=drop_prob,
                       init_final_bias=-3)
    block.append({'type': 'activation', 'activation_function': 'sigmoid', 'input_name': 'g', 'output_name': 'g'})

    # We now have h (block_input), h_hat, and g (using naming conventions from the paper)
    block.append({'type': 'merge_streams',
                  'input_name_1': 'g',
                  'input_name_2': 'h_hat',
                  'merge_type': 'multiply',
                  'output_name': 'first_term'})

    block.append({'type': 'merge_streams',
                  'input_name_1': 'one',  # We will need to give the model Tensor([1]) as _data['one']
                  'input_name_2': 'g',
                  'merge_type': 'subtract',
                  'output_name': 'one_minus_g'})

    block.append({'type': 'merge_streams',
                  'input_name_1': 'one_minus_g',
                  'input_name_2': 'block_input',
                  'merge_type': 'multiply',
                  'output_name': 'second_term'})

    block.append({'type': 'merge_streams',
                  'input_name_1': 'first_term',
                  'input_name_2': 'second_term',
                  'merge_type': 'add',
                  'output_name': output_name})

    return block