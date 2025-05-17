import torch
from torch import nn
import torch.nn.functional as F
from Bcramodel.S_LSTM.utils import CausalConv1D, BlockDiagonal


class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)

        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)

        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * (4/3)))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * (4/3)))
        self.down_proj = nn.Linear(int(hidden_size * (4/3)), hidden_size)

    def forward(self, x, prev_state):
        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        h_prev, c_prev, n_prev, m_prev = prev_state
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t

        output = self.down_proj(self.up_proj_left(h_t) * F.gelu(self.up_proj_right(h_t)))
        return output, (h_t, c_t, n_t, m_t)


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([sLSTMCell(input_size if i == 0 else hidden_size, hidden_size, num_heads)
                                     for i in range(num_layers)])

    def forward(self, x, state=None):
        # # ?υτ???
        model_device = next(self.parameters()).device
        # input_device = x.device
        #
        # # ????υτ???
        # print(f"Model device: {model_device}, Input device: {input_device}")
        #
        # if model_device != input_device:
        #     raise RuntimeError("Input tensors must be on the same device as the model.")

        seq_len, batch_size, _ = x.size()
        if state is None:
            state = [(torch.zeros(batch_size, self.hidden_size, device=model_device),
                       torch.zeros(batch_size, self.hidden_size, device=model_device),
                       torch.zeros(batch_size, self.hidden_size, device=model_device),
                       torch.zeros(batch_size, self.hidden_size, device=model_device)) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state[layer] = self.layers[layer](x_t, state[layer])
            outputs.append(x_t)

        # ????????????????
        return torch.stack(outputs), state