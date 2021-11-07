from torch import nn, einsum
from block_builder import Block
from RoPE_module import get_rope


class ConfigaFormer(nn.Module):
    def __init__(self,
                 model_config,
                 input_shapes,
                 ):
        super().__init__()
        # Type checking
        assert type(model_config['blocks']) == list, f"ConfigaFormer's config should be a list, it was given a {type(model_config)}"

        # create rope embeds
        self.rope_dict = get_rope(model_config['blocks'])
        self.input_shapes = input_shapes.copy()

        streams = input_shapes
        self.block_list = nn.ModuleList([])
        for i_block, _block in enumerate(model_config['blocks']):
            block_config = _block['config']
            N = _block['repeat']
            print(f"Block #{i_block+1}, {N}x")

            for n in range(N):
                if n == 0:
                    _print = True
                else:
                    _print = False

                block = Block(block_config=block_config, input_streams=streams, print_streams=_print)
                output_streams = block.streams
                streams = output_streams
                self.block_list.append(block)
            print("\n")

    def forward(self, _data):
        # Save input variable sizes in _data
        input_sizes = {}
        for stream_name in _data:
            assert stream_name in self.input_shapes.keys(), f"Stream name: {stream_name} was not in input stream names!"
            stream_shape_init = self.input_shapes[stream_name]
            stream_shape_received = _data[stream_name].shape

            for dim_idx in range(len(stream_shape_init)):
                dim_init = stream_shape_init[dim_idx]
                dim_received = stream_shape_received[dim_idx]

                if type(dim_init) == str:
                    input_sizes[dim_init] = dim_received

        _data['input_sizes'] = input_sizes

        # If rope embeds exist, then add frequency embeddings to _data
        if self.rope_dict != {}:
            # Get input device
            for stream_name in _data:
                device = _data[stream_name].device
                break

            for rope_key in self.rope_dict.keys():
                _data[rope_key] = self.rope_dict[rope_key].to(device)

        for i_block, _block in enumerate(self.block_list):
            _data = _block(_data)

        return _data




