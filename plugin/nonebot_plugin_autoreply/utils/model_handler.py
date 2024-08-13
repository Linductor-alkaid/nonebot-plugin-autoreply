import pickle
from pathlib import Path

def load_model(model_file: Path):
    """
    从指定的模型文件中加载模型。
    
    :param model_file: 模型文件的路径
    :return: 加载的模型对象
    """
    if model_file.exists():
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        model = {}  # 如果模型文件不存在，初始化一个空模型
    return model

def save_model(model: dict, model_file: Path):
    """
    将模型保存到指定的文件中。
    
    :param model: 要保存的模型对象，通常是一个字典
    :param model_file: 模型文件的路径
    """
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

def update_model(new_data: dict, model: dict):
    """
    根据新的数据更新模型。
    
    :param new_data: 新的训练数据，格式为字典
    :param model: 现有的模型对象
    :return: 更新后的模型对象
    """
    # 简单示例：根据新数据更新模型中的用户消息计数
    user_id = new_data['user_id']
    if user_id in model:
        model[user_id]['message_count'] += 1
    else:
        model[user_id] = {'message_count': 1}
    return model
