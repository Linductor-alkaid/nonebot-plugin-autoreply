import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# 定义模型文件路径
model_file = Path(__file__).parent.parent / "data" / "model" / "gpt2_model"
tokenizer_dir = Path(__file__).parent.parent / "data" / "model" / "gpt2_tokenizer"

def load_gpt_model():
    """
    加载 GPT 模型和分词器。如果模型文件不存在，初始化一个新的 GPT 模型。
    
    :return: GPT 模型和分词器
    """
    if model_file.exists() and tokenizer_dir.exists():
        model = GPT2LMHeadModel.from_pretrained(model_file)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    else:
        # 如果模型文件不存在，加载预训练的 GPT-2 模型
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer

def save_gpt_model(model):
    """
    将训练后的 GPT 模型保存到文件。
    
    :param model: 训练后的 GPT 模型
    """
    model.save_pretrained(model_file)

def update_model(new_data: dict, model, tokenizer):
    """
    根据新的数据更新 GPT 模型。
    
    :param new_data: 新的训练数据，格式为字典
    :param model: 当前的 GPT 模型
    :param tokenizer: GPT 模型的分词器
    :return: 更新后的模型
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 构建训练输入，包含用户的消息和与上次消息的时间差
    input_text = f"User said: {new_data['message']} " \
                 f"Time since last message: {new_data['time_diff']} seconds."

    inputs = tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True)
    inputs = inputs.to(model.device)  # 将输入数据移动到同一个设备（GPU或CPU）
    
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return model
