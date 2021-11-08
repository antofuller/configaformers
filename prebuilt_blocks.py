

def get_transformer_block(num_heads,
                          dim,
                          main_stream='x',
                          ff_mult=4,
                          use_attn_offset=True,
                          offset_name='attn_offset',
                          drop_prob=0,
                          ):
    block = [] # Make list

    # Make residual and norm
    block.append({'type': 'make_stream', 'input_name': main_stream, 'output_name': 'residual'})
    block.append({'type': 'norm', 'input_name': main_stream, 'output_name': main_stream, 'norm_type': 'layer_norm'})

    # Make QKVs
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': 'queries'})
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': 'keys'})
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': 'values'})

    block.append({'type': 'make_heads', 'input_name': 'queries', 'output_name': 'queries', 'num_heads': num_heads})
    block.append({'type': 'make_heads', 'input_name': 'keys', 'output_name': 'keys', 'num_heads': num_heads})
    block.append({'type': 'make_heads', 'input_name': 'values', 'output_name': 'values', 'num_heads': num_heads})

    # Perform attention
    block.append({'type': 'mha_dots',
                  'input_name_queries': 'queries',
                  'input_name_keys': 'keys',
                  'output_name': 'attn_dots'})
    if use_attn_offset:
        block.append({'type': 'attention_offset',
                      'input_name_attn_dots': 'attn_dots',
                      'input_name_attn_offset': offset_name,
                      'output_name': 'attn_dots'})
    block.append({'type': 'mha_sum',
                  'input_name_values': 'values',
                  'input_name_attn_dots': 'attn_dots',
                  'output_name': main_stream})
    block.append({'type': 'merge_heads', 'input_name': main_stream, 'output_name': main_stream})

    # Mix
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': main_stream})

    # Add residual
    block.append({'type': 'merge_streams',
                  'input_name_1': main_stream,
                  'input_name_2': 'residual',
                  'merge_type': 'add',
                  'output_name': main_stream})

    # Make residual and norm
    block.append({'type': 'make_stream', 'input_name': main_stream, 'output_name': 'residual'})
    block.append({'type': 'norm', 'input_name': main_stream, 'output_name': main_stream, 'norm_type': 'layer_norm'})

    # Proj Up
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': main_stream, 'output_dim': dim * ff_mult})

    # Activation
    block.append({'type': 'activation', 'input_name': main_stream, 'output_name': main_stream})

    if drop_prob != 0:
        # Dropout
        block.append({'type': 'dropout', 'prob': drop_prob, 'input_name': main_stream, 'output_name': main_stream})

    # Proj Down
    block.append({'type': 'linear', 'input_name': main_stream, 'output_name': main_stream, 'output_dim': dim})

    # Add residual
    block.append({'type': 'merge_streams',
                  'input_name_1': main_stream,
                  'input_name_2': 'residual',
                  'merge_type': 'add',
                  'output_name': main_stream})

    return block
