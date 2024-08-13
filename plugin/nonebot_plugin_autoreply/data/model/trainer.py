import pickle
from pathlib import Path
import json

# 定义模型文件路径
model_file = Path(__file__).parent / "chat_model.pkl"

def train_model(data_file: Path):
    """
    从聊天数据文件中加载数据并训练模型，然后保存更新后的模型。
    
    :param data_file: 存储聊天数据的文件路径
    """
    # 加载现有模型
    model = load_model()

    # 从数据文件中加载聊天数据
    with open(data_file, 'r') as f:
        chat_data = json.load(f)

    # 模拟的训练逻辑（这里需要根据实际需求进行实现）
    for entry in chat_data:
        user_id = entry['user_id']
        message = entry['message']

        # 简单示例：记录每个用户的消息数量
        if user_id in model:
            model[user_id]['message_count'] += 1
        else:
            model[user_id] = {'message_count': 1}

    # 保存更新后的模型
    save_model(model)

def load_model():
    """
    加载模型文件中的数据。
    
    :return: 加载的模型数据
    """
    if model_file.exists() and model_file.stat().st_size > 0:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        except EOFError:
            # 如果文件内容为空，初始化一个空模型
            print("模型文件为空，初始化空模型")
            model = {}
    else:
        model = {}  # 如果模型文件不存在或为空，初始化一个空模型
    return model


def save_model(model):
    """
    将训练后的模型数据保存到文件。
    
    :param model: 要保存的模型数据
    """
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
