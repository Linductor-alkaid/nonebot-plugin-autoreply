import os
import json
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# 定义模型和分词器的文件路径
model_dir = Path(__file__).parent / "gpt2_model"
tokenizer_dir = Path(__file__).parent / "gpt2_tokenizer"
pretrain_data_folder = Path(__file__).parent.parent / "data" / "pretrain"

# 确定设备为 GPU，如果不可用则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pretrain_model():
    """
    使用大量文本数据对 GPT 模型进行预训练。
    """
    # 加载模型和分词器，并将模型移至 GPU
    model, tokenizer = load_model()
    model.to(device)

    # 收集所有文本数据
    texts = []
    for file_name in os.listdir(pretrain_data_folder):
        with open(pretrain_data_folder / file_name, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    # 训练 GPT 模型
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for text in texts:
        inputs = tokenizer.encode(text, return_tensors='pt').to(device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 保存预训练后的模型
    save_model(model, tokenizer)

def train_model(data_file: Path):
    """
    从聊天数据文件中加载数据并训练GPT模型，然后保存更新后的模型。
    
    :param data_file: 存储聊天数据的文件路径
    """
    # 加载现有模型，并将模型移至 GPU
    model, tokenizer = load_model()
    model.to(device)

    # 从数据文件中加载聊天数据
    with open(data_file, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)

    # 模拟的训练逻辑
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for entry in chat_data:
        user_id = entry.get('user_id', 'unknown')
        message = entry.get('message', '')
        time_diff = entry.get('time_diff', 0)  # 设置默认值为 0
        
        # 构建训练输入，包含用户的消息和与上次消息的时间差
        input_text = f"User said: {message} " \
                     f"Time since last message: {time_diff} seconds."
        inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # 保存更新后的模型
    save_model(model, tokenizer)

def load_model():
    """
    加载 GPT 模型和分词器。如果模型文件不存在，初始化一个新的 GPT 模型。
    
    :return: GPT 模型和分词器
    """
    if model_dir.exists() and tokenizer_dir.exists():
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    else:
        print("模型或分词器文件夹不存在，初始化空模型")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    return model, tokenizer

def save_model(model, tokenizer):
    """
    将训练后的 GPT 模型保存到文件。
    
    :param model: 训练后的 GPT 模型
    :param tokenizer: GPT 模型的分词器
    """
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
