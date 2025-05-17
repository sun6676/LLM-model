import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel, MambaForCausalLM
import time
from config import get_config
from data import load_dataset
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, TextCNN_Model, Transformer_CNN_RNN, \
    Transformer_Attention, Transformer_CNN_RNN_Attention, Transformer_CNN_RNN_MultiHeadAttention, \
    Transformer_CNN_XLSTM_Attention, Transformer_3Dcnn_XLSTM, Transformer_XLSTM, Transformer_CNN_BiLSTM, \
    Transformer_CNN_BiLSTM_att, Transformer_CNN_BiLSTM_BiGRU
import matplotlib

matplotlib.use('TkAgg')


class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        # 创建模型
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('C:/2024Project/BertProject1/bert-base-chinese')
            self.input_size = 768
            base_model = AutoModel.from_pretrained('C:/2024Project/BertProject1/bert-base-chinese')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('C:/2024Project/BertProject1/roberta-base',
                                                           add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('C:/2024Project/BertProject1/roberta-base')
        elif args.model_name == 'lert':
            self.tokenizer = AutoTokenizer.from_pretrained('C:/2024Project/BertProject1/lert-base-chinese',
                                                           add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('C:/2024Project/BertProject1/lert-base-chinese')
        elif args.model_name == 'pert':
            self.tokenizer = AutoTokenizer.from_pretrained('C:/2024Project/BertProject1/pert-base-chinese',
                                                           add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('C:/2024Project/BertProject1/pert-base-chinese')
        elif args.model_name == 'mamba2':
            self.tokenizer = AutoTokenizer.from_pretrained('/Mamba2', add_prefix_space=True)
            self.input_size = 768
            base_model = MambaForCausalLM.from_pretrained('/Mamba2')
        elif args.model_name == 'mamba-370m':
            self.tokenizer = AutoTokenizer.from_pretrained('C:/2024Project/BertProject1/Mamba-370m', add_prefix_space=True)
            self.input_size = 768
            base_model = MambaForCausalLM.from_pretrained('C:/2024Project/BertProject1/Mamba-370m')
        elif args.model_name == 'mamba2-780m':  #有bug
            self.tokenizer = AutoTokenizer.from_pretrained('C:/2024Project/BertProject1/Mamba2-780m', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('C:/2024Project/BertProject1/Mamba2-780m')
        else:
            raise ValueError('unknown model')

        # 根据 method_name 选择模型
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'textcnn':
            self.Mymodel = TextCNN_Model(base_model, args.num_classes)
        elif args.method_name == 'attention':
            self.Mymodel = Transformer_Attention(base_model, args.num_classes)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        elif args.method_name == 'lstm_textcnn_attention':
            self.Mymodel = Transformer_CNN_RNN_Attention(base_model, args.num_classes)
        elif args.method_name == 'lstm_textcnn_muliattention':
            self.Mymodel = Transformer_CNN_RNN_MultiHeadAttention(base_model, args.num_classes)
        elif args.method_name == 'xlstm_textcnn_attention':
            self.Mymodel = Transformer_CNN_XLSTM_Attention(base_model, args.num_classes)
        elif args.method_name == 'xlstm_3dcnn':
            self.Mymodel = Transformer_3Dcnn_XLSTM(base_model, args.num_classes)
        elif args.method_name == 'xlstm':
            self.Mymodel = Transformer_XLSTM(base_model, args.num_classes)
        elif args.method_name == 'cnn-bilstm':
            self.Mymodel = Transformer_CNN_BiLSTM(base_model, args.num_classes)
        elif args.method_name == 'cnn-bilstm-att':
            self.Mymodel = Transformer_CNN_BiLSTM_att(base_model, args.num_classes)
        elif args.method_name == 'cnn-bilstm-gru':
            self.Mymodel = Transformer_CNN_BiLSTM_BiGRU(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        # 将模型放置到设备上
        self.Mymodel.to(self.args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _calculate_metrics(self, y_true, y_pred):
        # 安全地计算指标，避免除零错误
        try:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            precision, recall, f1 = 0, 0, 0
        return precision, recall, f1

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        y_true_train, y_pred_train = [], []

        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

            y_true_train.extend(targets.cpu().numpy())
            y_pred_train.extend(torch.argmax(predicts, dim=1).cpu().numpy())

        train_precision, train_recall, train_f1 = self._calculate_metrics(y_true_train, y_pred_train)
        return train_loss / n_train, n_correct / n_train, train_precision, train_recall, train_f1

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        y_true_test, y_pred_test = [], []
        # ////////////
        all_features = []  # 保存所有特征
        all_targets = []  # 保存所有标签
        # ////////////
        self.Mymodel.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)

                # Get predictions
                predicts = self.Mymodel(inputs)
                # /////////////////////////////////////////
                # Get feature vector before classification layer
                raw_outputs = self.Mymodel.base_model(**inputs)
                tokens = raw_outputs.last_hidden_state
                cnn_tokens = tokens.unsqueeze(1)
                cnn_out = torch.cat([self.Mymodel.conv_pool(cnn_tokens, conv) for conv in self.Mymodel.convs], 1)
                rnn_outputs, _ = self.Mymodel.lstm(tokens)
                rnn_out = rnn_outputs[:, -1, :]
                combined_out = torch.cat((cnn_out, rnn_out), 1)  # 这就是最终的特征向量

                # 保存特征和目标
                all_features.append(combined_out.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                # //////////////////////////////////////////////////////
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

                y_true_test.extend(targets.cpu().numpy())
                y_pred_test.extend(torch.argmax(predicts, dim=1).cpu().numpy())

        test_precision, test_recall, test_f1 = self._calculate_metrics(y_true_test, y_pred_test)
        # 将所有特征和目标拼接
        all_features = np.concatenate(all_features, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        return test_loss / n_test, n_correct / n_test, test_precision, test_recall, test_f1 , all_features, all_targets

    def run(self):
        try:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            self.logger.info(f"尝试从检查点继续训练: {checkpoint_path}")
            self.Mymodel.load_state_dict(torch.load(checkpoint_path))
        except FileNotFoundError:
            self.logger.warning("没有找到检查点文件，将从头开始训练。")

        self.logger.info(f'CUDA 已分配内存: {torch.cuda.memory_allocated(self.args.device.index)} bytes')
        self.logger.info(f'CUDA 已保留内存: {torch.cuda.memory_reserved(self.args.device.index)} bytes')
        # Print the parameters of model
        # for name, layer in self.Mymodel.named_parameters(recurse=True):
        # print(name, layer.shape, sep=" ")
        train_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        start_time = time.time()  # 记录训练开始时间
        l_acc, l_trloss, l_teloss, l_epo, l_train_f1, l_test_f1 = [], [], [], [], [], []
        l_train_precision, l_train_recall, l_test_precision, l_test_recall = [], [], [], []
        all_test_features = []
        all_test_targets = []
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc, train_precision, train_recall, train_f1 = self._train(train_dataloader, criterion,
                                                                                         optimizer)
            # 测试并收集特征
            test_loss, test_acc, test_precision, test_recall, test_f1, test_features, test_targets = self._test(test_dataloader, criterion)

            all_test_features.append(test_features)
            all_test_targets.append(test_targets)
            l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
            l_train_f1.append(train_f1), l_test_f1.append(test_f1)
            l_train_precision.append(train_precision), l_train_recall.append(train_recall)
            l_test_precision.append(test_precision), l_test_recall.append(test_recall)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                checkpoint_name = f'best_model_epoch{epoch + 1}_acc{test_acc * 100:.2f}_loss{test_loss:.4f}.pth'
                checkpoint_path = os.path.join(self.args.checkpoint_dir, checkpoint_name)
                torch.save(self.Mymodel.state_dict(), checkpoint_path)
                self.logger.info(f'新最佳模型已保存于: {checkpoint_path}')
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info(
                '[train] loss: {:.4f}, acc: {:.2f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(train_loss,
                                                                                                          train_acc * 100,
                                                                                                          train_precision,
                                                                                                          train_recall,
                                                                                                          train_f1))
            self.logger.info(
                '[test] loss: {:.4f}, acc: {:.2f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(test_loss,
                                                                                                         test_acc * 100,
                                                                                                         test_precision,
                                                                                                         test_recall,
                                                                                                         test_f1))
        end_time = time.time()  # 记录训练结束时间
        training_time = end_time - start_time  # 计算训练时间
        self.logger.info('Training time: {:.2f} seconds'.format(training_time))  # 记录训练时间
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        # # tsne合并目标
        all_test_features = np.concatenate(all_test_features, axis=0)
        all_test_targets = np.concatenate(all_test_targets, axis=0)
        # 绘制t-SNE图
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_test_features)
        #
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_test_targets, cmap='viridis', s=10)
        plt.xlabel('', fontsize=30)  # 尽管这里设置了fontsize，但对于空字符串，它不会影响字体大小
        plt.ylabel('', fontsize=30)  # 同上
        plt.title("WB1 Data Feature representation", fontsize=20)
        # 修改坐标轴上的字体大小
        ax = plt.gca()  # 获取当前坐标轴
        ax.tick_params(axis='both', which='major', labelsize=16)  # 设定主刻度标签的字体大小为20
        ax.tick_params(axis='both', which='minor', labelsize=16)  # 如果有需要，也可以设定次刻度标签的字体大小
        # #plt.colorbar()  # 显示颜色条
        plt.savefig('./output/wb1t-sne.png')
        plt.show()
        # Draw the training process
        plt.plot(l_epo, l_acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('./output/acc.png')

        plt.plot(l_epo, l_teloss)
        plt.ylabel('test-loss')
        plt.xlabel('epoch')
        plt.savefig('./output/teloss.png')

        plt.plot(l_epo, l_trloss)
        plt.ylabel('train-loss')
        plt.xlabel('epoch')
        plt.savefig('./output/trloss.png')


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = Niubility(args, logger)
    # print(torch.__version__)  # 确认 PyTorch 版本
    # print(torch.cuda.is_available())  # True 表示 GPU 可用
    # print(torch.version.cuda)  # 查看 CUDA 版本
    # print(torch.cuda.get_device_name(0))  # 打印 GPU 名称
    nb.run()
