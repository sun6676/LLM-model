import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from prettytable import PrettyTable
from Bcramodel.model import Transformer_CNN_RNN_Attention

# 加载预训练的BERT模型和分词器
base_model = BertModel.from_pretrained('../bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

# Initialize your model
model = Transformer_CNN_RNN_Attention(base_model, num_classes=2)

# Load the trained model weights
model.load_state_dict(torch.load('./checkpoints/best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(inputs)
    probabilities = F.softmax(outputs, dim=1)
    return probabilities


print("请输入文本进行情感分析，输入 'q' 来结束预测")
user_inputs = []  # 用于存储所有输入的文本
results = []  # 用于存储所有预测结果

while True:
    print("请在此输入：")
    user_input = input()
    if user_input.lower() == 'q':
        break
    elif len(user_input) < 4:
        print("输入的文本太短，请至少输入4个字符的文本。")
        continue
    probabilities = predict_sentiment(user_input)
    positive_probability = probabilities[0][1].item()
    negative_probability = probabilities[0][0].item()
    user_inputs.append(user_input)
    results.append((positive_probability, negative_probability))
    print("\n当前文本的情感分析结果：")
    print(f"积极概率: {positive_probability:.2%}")
    print(f"消极概率: {negative_probability:.2%}")

if user_input.lower() == 'q':
    # 使用PrettyTable展示所有结果，包括情感标签
    table = PrettyTable(["序号", "文本", "积极概率", "消极概率", "情感"])

    for i, (text, (pos_prob, neg_prob)) in enumerate(zip(user_inputs, results), start=1):
        sentiment_label = "积极" if pos_prob > neg_prob else "消极"
        table.add_row([i, text, f"{pos_prob:.2%}", f"{neg_prob:.2%}", sentiment_label])

    print("\n\n预测结果表格：")
    print(table)