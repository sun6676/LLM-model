import torch
from torch import nn

from Bcramodel.S_LSTM.Mlstm import mLSTMBlock
from Bcramodel.S_LSTM.Slstm import sLSTMCell


class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, layers, batch_first=False,
                 proj_factor_mlstm=2):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.num_layers = len(layers)
        self.batch_first = batch_first
        self.proj_factor_mlstm = proj_factor_mlstm

        self.layer_list = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMCell(input_size if len(self.layer_list) == 0 else hidden_size,
                                   hidden_size, num_heads)
            elif layer_type == 'm':
                layer = mLSTMBlock(input_size if len(self.layer_list) == 0 else hidden_size,
                                    hidden_size, num_heads, proj_factor_mlstm)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layer_list.append(layer)

    def forward(self, x, state=None):
        # # 设备检查
        model_device = next(self.parameters()).device
        # input_device = x.device
        #
        # # 打印设备信息
        # print(f"Model device: {model_device}, Input device: {input_device}")
        #
        # if model_device != input_device:
        #     raise RuntimeError("Input tensors must be on the same device as the model.")

        assert x.ndim == 3
        if self.batch_first:
            x = x.transpose(0, 1)  # seq_len, batch_size, input_size
        seq_len, batch_size, _ = x.size()

        if state is not None:
            state = torch.stack(list(state))
            assert state.ndim == 4
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4
            assert state_num_layers == self.num_layers
            assert state_batch_size == batch_size
            assert state_input_size == self.hidden_size  # 修改为 hidden_size
            state = state.transpose(0, 1)
        else:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=model_device)

        output = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layer_list[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))
            output.append(x_t)

        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)  # batch_size, seq_len, hidden_size
        state = tuple(state.transpose(0, 1))
        return output, state