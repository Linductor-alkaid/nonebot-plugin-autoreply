import json
import pickle
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# 定义模型文件路径
model_file = Path(__file__).parent / "chat_model.pkl"

def train_model(data_file: Path):
    """
    从聊天数据文件中加载数据并训练GPT模型，然后保存更新后的模型。
    
    :param data_file: 存储聊天数据的文件路径
    """
    # 加载现有模型
    model, tokenizer = load_model()

    # 从数据文件中加载聊天数据
    with open(data_file, 'r') as f:
        chat_data = json.load(f)

    # 模拟的训练逻辑（这里需要根据实际需求进行实现）
    training_data = []
    for entry in chat_data:
        user_id = entry['user_id']
        message = entry['message']
        time_diff = entry['time_diff']
        
        # 构建训练输入，包含用户的消息和与上次消息的时间差
        input_text = f"User said: {message} " \
                     f"Time since last message: {time_diff} seconds."
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        training_data.append(inputs)
    
    # 训练 GPT 模型
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    for inputs in training_data:
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
    if model_file.exists() and model_file.stat().st_size > 0:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        except EOFError:
            print("模型文件为空，初始化空模型")
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    return model, tokenizer

def save_model(model, tokenizer):
    """
    将训练后的 GPT 模型和分词器保存到文件。
    
    :param model: 训练后的 GPT 模型
    :param tokenizer: GPT 模型的分词器
    """
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    # 注意：tokenizer 不需要保存，通常可以从预训练模型中重新加载
