import math

import torch
import torch.nn.functional as F
from torch import nn

from Bcramodel.S_LSTM.Xlstm import xLSTM


# Bert + FNN
class Transformer(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        # The pooler_output is made of CLS --> FNN --> Tanh
        # The last_hidden_state[:,0] is made of original CLS
        # Method one
        # cls_feats  = raw_outputs.pooler_output
        # Method two
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        predicts = self.softmax(self.linear(self.dropout(cls_feats)))
        return predicts


class Gru_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Gru = nn.GRU(input_size=self.input_size,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        gru_output, _ = self.Gru(tokens)
        outputs = gru_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


# Try to use the softmax、relu、tanh and logistic
class Lstm_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=320,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_output, _ = self.Lstm(tokens)
        outputs = lstm_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


class BiLstm_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        # Open the bidirectional
        self.BiLstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=320,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320 * 2, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        outputs, _ = self.BiLstm(cls_feats)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


class Rnn_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        outputs, _ = self.Rnn(cls_feats)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


class TextCNN_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.encode_layer = 12

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        tokens = conv(tokens)
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)
        tokens = F.max_pool1d(tokens, tokens.size(2))
        out = tokens.squeeze(2)
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)
        out = torch.cat([self.conv_pool(tokens, conv) for conv in self.convs],
                        1)
        predicts = self.block(out)
        return predicts


class Transformer_CNN_RNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cnn_tokens = raw_outputs.last_hidden_state.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]
        rnn_tokens = raw_outputs.last_hidden_state
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts


class Transformer_Attention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)
        attention_output = torch.mean(attention_output, dim=1)

        predicts = self.block(attention_output)
        return predicts


class Transformer_CNN_RNN_Attention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)
        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        # Self-Attention
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)

        # TextCNN
        cnn_tokens = attention_output.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]

        rnn_tokens = tokens
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts


class Transformer_CNN_RNN_MultiHeadAttention(nn.Module):
    def __init__(self, base_model, num_classes, num_heads=8):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = True

        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.base_model.config.hidden_size,
                                                    num_heads=num_heads,
                                                    batch_first=True)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(query=tokens, key=tokens, value=tokens)

        # TextCNN
        cnn_tokens = attn_output.unsqueeze(1)
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs], 1)

        # LSTM
        rnn_tokens = attn_output  # 使用多头注意力的输出作为LSTM的输入
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]

        # Combine CNN and LSTM outputs
        combined_out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(combined_out)
        return predicts


class Transformer_CNN_XLSTM_Attention(nn.Module):
    def __init__(self, base_model, num_classes,  num_heads=4):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # 使用 xLSTM
        self.xlstm = xLSTM(input_size=self.base_model.config.hidden_size, hidden_size=512,
                           num_heads=num_heads, layers=['s', 'm'])
        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        # Self-Attention
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)

        # TextCNN
        cnn_tokens = attention_output.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]

        rnn_tokens = tokens
        rnn_outputs, _ = self.xlstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts


class Transformer_3Dcnn_XLSTM(nn.Module):
    def __init__(self, base_model, num_classes,  num_heads=4):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = True

        # 3D CNN Hyperparameters
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(5, 5, 5), padding=2, stride=(8, 8, 8)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 5), padding=2, stride=(2, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(5, 5, 5), padding=2),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(5, 5, 5), padding=2)
        ])

        # Attention layer
        self.attention = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(5, 5, 5), padding=2)

         # Max Pooling layer
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 1))

        # Adaptive pooling to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 使用 xLSTM
        self.xlstm = xLSTM(input_size=self.base_model.config.hidden_size, hidden_size=512,
                           num_heads=num_heads, layers=['s', 'm'])

        # Define the fully connected block
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 + 512, 128),  # 64 from CNN (after adaptive pooling), 512 from LSTM
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),  # For multi-class classification
            nn.Softmax(dim=1)  # Softmax activation for multi-class classification
        )

    def conv_pool(self, tokens, conv):
        # Apply a single 3D Convolutional layer and max pooling
        x = conv(tokens)
        x = F.relu(x)
        x = self.max_pool(x)
        return x

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        # Reshape tokens for 3D CNN
        tokens_reshaped = tokens.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, hidden_size, 1, seq_len]

        # Apply 3D Convolutional layers and max pooling
        x = tokens_reshaped
        for conv in self.convs:
            x = self.conv_pool(x, conv)  # Call conv_pool for each conv layer

        # Apply attention mechanism
        attention = torch.sigmoid(self.attention(x))
        attention = F.interpolate(attention, size=(x.size(2), x.size(3), x.size(4)), mode='trilinear',
                                  align_corners=False)

        # Apply attention
        x = x * attention  # Element-wise multiplication

        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        # 使用 xLSTM
        rnn_outputs, _ = self.xlstm(tokens)  # tokens 作为输入
        rnn_out = rnn_outputs[:, -1, :]  # 取最后一个时间步的输出
        # print(f"rnn_out shape: {rnn_out.shape}")

        # 输出处理
        # Concatenate CNN and RNN outputs
        out = torch.cat((x, rnn_out), dim=1)
        predicts = self.block(out)
        return predicts


class Transformer_XLSTM(nn.Module):
    def __init__(self, base_model, num_classes, num_heads=4):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = True

        # 使用 xLSTM
        self.xlstm = xLSTM(input_size=self.base_model.config.hidden_size, hidden_size=512,
                           num_heads=num_heads, layers=['s', 'm'])

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),  # 输出大小应为 hidden_size
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        # # 检查输入的设备是否与模型设备一致
        # if next(self.parameters()).device != inputs['input_ids'].device:
        #     raise RuntimeError("Input tensors must be on the same device as the model.")

        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        # 使用 xLSTM
        rnn_outputs, _ = self.xlstm(tokens)  # tokens 作为输入
        rnn_out = rnn_outputs[:, -1, :]  # 取最后一个时间步的输出

        # 输出处理
        out = rnn_out  # 输出为最后一个时间步的输出
        predicts = self.block(out)
        return predicts


class Transformer_CNN_BiLSTM(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = True

        # 定义超参数
        self.num_filters = 100  # 卷积核数量
        self.kernel_size = 3   # 卷积核大小

        # CNN
        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_filters,
            kernel_size=(self.kernel_size, self.base_model.config.hidden_size)
        )

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.base_model.config.hidden_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # 双向
        )

        # 分类器
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(100 + 512, 128),  # 100 是 CNN 输出，512 是 BiLSTM 双向输出
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        # 获取预训练模型的输出
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]

        # CNN 模块
        cnn_tokens = tokens.unsqueeze(1)  # 增加维度: shape [batch_size, 1, seq_len, hidden_size]
        cnn_out = self.cnn(cnn_tokens)  # shape: [batch_size, num_filters, seq_len - kernel_size + 1, 1]
        cnn_out = F.relu(cnn_out)
        cnn_out = cnn_out.squeeze(3)  # shape: [batch_size, num_filters, seq_len - kernel_size + 1]
        cnn_out = F.max_pool1d(cnn_out, cnn_out.size(2))  # 全局最大池化
        cnn_out = cnn_out.squeeze(2)  # shape: [batch_size, num_filters]

        # BiLSTM 模块
        bilstm_out, _ = self.bilstm(tokens)  # shape: [batch_size, seq_len, hidden_size * 2]
        bilstm_out = bilstm_out[:, -1, :]    # 取最后时间步的输出: shape [batch_size, hidden_size * 2]

        # 拼接 CNN 和 BiLSTM 的输出
        combined_out = torch.cat((cnn_out, bilstm_out), dim=1)  # shape: [batch_size, num_filters + hidden_size * 2]

        # 全连接分类器
        predicts = self.block(combined_out)
        return predicts

class Transformer_CNN_BiLSTM_att(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = True

        # 定义超参数
        self.num_filters = 100  # 卷积核数量
        self.kernel_size = 3   # 卷积核大小

        # CNN
        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_filters,
            kernel_size=(self.kernel_size, self.base_model.config.hidden_size)
        )

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.base_model.config.hidden_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # 双向
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(512, 128),  # 512 是 BiLSTM 双向输出的维度
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(100 + 512, 128),  # 100 是 CNN 输出，512 是 BiLSTM+注意力的输出
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        # 获取预训练模型的输出
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]

        # CNN 模块
        cnn_tokens = tokens.unsqueeze(1)  # 增加维度: shape [batch_size, 1, seq_len, hidden_size]
        cnn_out = self.cnn(cnn_tokens)  # shape: [batch_size, num_filters, seq_len - kernel_size + 1, 1]
        cnn_out = F.relu(cnn_out)
        cnn_out = cnn_out.squeeze(3)  # shape: [batch_size, num_filters, seq_len - kernel_size + 1]
        cnn_out = F.max_pool1d(cnn_out, cnn_out.size(2))  # 全局最大池化
        cnn_out = cnn_out.squeeze(2)  # shape: [batch_size, num_filters]

        # BiLSTM 模块
        bilstm_out, _ = self.bilstm(tokens)  # shape: [batch_size, seq_len, hidden_size * 2]

        # 注意力机制
        attention_weights = self.attention(bilstm_out)  # shape: [batch_size, seq_len, 1]
        attention_weights = attention_weights.squeeze(2)  # shape: [batch_size, seq_len]
        attention_weights = attention_weights.unsqueeze(1)  # shape: [batch_size, 1, seq_len]
        context_vector = torch.bmm(attention_weights, bilstm_out)  # shape: [batch_size, 1, hidden_size * 2]
        context_vector = context_vector.squeeze(1)  # shape: [batch_size, hidden_size * 2]

        # 拼接 CNN 和 BiLSTM + 注意力的输出
        combined_out = torch.cat((cnn_out, context_vector), dim=1)  # shape: [batch_size, num_filters + hidden_size * 2]

        # 全连接分类器
        predicts = self.block(combined_out)
        return predicts


class Transformer_CNN_RNN_Attention_one(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = True

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)

        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        # Final classification block
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),  # Adjusted input size after concatenating CNN and LSTM outputs
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # Apply convolution and pooling
        tokens = conv(tokens)  # shape [batch_size, out_channels, L-K+1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, L-K+1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape [batch_size, out_channels, 1]
        out = tokens.squeeze(2)  # shape [batch_size, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state  # shape [batch_size, seq_len, hidden_size]

        # Self-Attention
        K = self.key_layer(tokens)  # shape [batch_size, seq_len, hidden_size]
        Q = self.query_layer(tokens)  # shape [batch_size, seq_len, hidden_size]
        V = self.value_layer(tokens)  # shape [batch_size, seq_len, hidden_size]

        # Compute attention scores
        attention_scores = torch.bmm(Q, K.permute(0, 2, 1)) * self._norm_fact  # shape [batch_size, seq_len, seq_len]
        attention = F.softmax(attention_scores, dim=-1)  # shape [batch_size, seq_len, seq_len]

        # Compute attention-weighted output
        attention_output = torch.bmm(attention, V)  # shape [batch_size, seq_len, hidden_size]

        # TextCNN
        cnn_tokens = attention_output.unsqueeze(1)  # shape [batch_size, 1, seq_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape [batch_size, num_filters * len(filter_sizes)]

        # Apply LSTM
        rnn_tokens = tokens
        rnn_outputs, _ = self.lstm(rnn_tokens)  # LSTM layer output
        rnn_out = rnn_outputs[:, -1, :]  # Get the last hidden state from LSTM (shape [batch_size, hidden_size * 2])

        # Concatenate CNN and LSTM outputs
        out = torch.cat((cnn_out, rnn_out), 1)  # shape [batch_size, cnn_out_size + rnn_out_size]

        # Apply the final classification block
        predicts = self.block(out)  # shape [batch_size, num_classes]
        return predicts

class Transformer_CNN_BiLSTM_BiGRU(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Set requires_grad for base model parameters
        for param in base_model.parameters():
            param.requires_grad = True

        # Define the hyperparameters for CNN
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN: Create convolutional layers with different filter sizes
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # BiLSTM (Bidirectional LSTM)
        self.bilstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                              hidden_size=512,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)

        # BiGRU (Bidirectional GRU)
        self.bigru = nn.GRU(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        # Block for final classification
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2348, 128),  # Adjusted input size after concatenating CNN, BiLSTM, and BiGRU outputs
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)  # Output class probabilities (for multi-class classification)
        )

    def conv_pool(self, tokens, conv):
        # Apply the convolutional layer and pooling
        tokens = conv(tokens)  # shape [batch_size, out_channels, text_length, 1]
        tokens = F.relu(tokens)  # Apply ReLU activation
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, text_length]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape [batch_size, out_channels, 1]
        out = tokens.squeeze(2)  # shape [batch_size, out_channels]
        return out

    def forward(self, inputs):
        # Pass the inputs through the base transformer model
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state  # shape [batch_size, seq_len, hidden_size]

        # Apply TextCNN (use different filter sizes)
        cnn_tokens = tokens.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape [batch_size, num_filters * len(self.filter_sizes)]

        # Apply BiLSTM (Bidirectional LSTM)
        rnn_tokens = tokens
        bilstm_outputs, _ = self.bilstm(rnn_tokens)  # BiLSTM layer
        bilstm_out = bilstm_outputs[:, -1, :]  # Get the last hidden state from BiLSTM (shape [batch_size, hidden_size * 2])

        # Apply BiGRU (Bidirectional GRU)
        bigru_outputs, _ = self.bigru(rnn_tokens)  # BiGRU layer
        bigru_out = bigru_outputs[:, -1, :]  # Get the last hidden state from BiGRU (shape [batch_size, hidden_size * 2])

        # Concatenate the CNN, BiLSTM, and BiGRU outputs
        out = torch.cat((cnn_out, bilstm_out, bigru_out), 1)  # shape [batch_size, cnn_out_size + bilstm_out_size + bigru_out_size]

        # Apply the final classification block
        predicts = self.block(out)  # Final classification output
        return predicts
