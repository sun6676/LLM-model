import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
import torch


def get_config():
    parser = argparse.ArgumentParser()
    '''Base'''

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='lert',
                        choices=['bert', 'roberta', 'lert', 'pert', 'mamba2', 'mamba-370m', 'mamba2-780m'])
    parser.add_argument('--method_name', type=str, default='lstm_textcnn_attention',
                        choices=['gru', 'rnn', 'bilstm', 'lstm', 'fnn', 'textcnn', 'attention', 'lstm+textcnn',
                                 'lstm_textcnn_attention', 'lstm_textcnn_muliattention', 'bilstm_textcnn_attention',
                                 'xlstm', 'xlstm_textcnn_attention', 'xlstm_3dcnn','cnn-bilstm','cnn-bilstm-att',
                                 'cnn-bilstm-gru'])

    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    '''save-model'''
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Directory to save model checkpoints. Default is 'checkpoints'.")

    args = parser.parse_args()
    args.device = torch.device(args.device)

    # 保存模型路径
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    '''logger'''
    args.log_name = '{}_{}_{}.log'.format(args.model_name, args.method_name,
                                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
