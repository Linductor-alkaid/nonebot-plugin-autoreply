import json
from pathlib import Path

def save_chat_data(chat_data: dict, data_file: Path):
    """
    将新的聊天数据保存到指定的 JSON 文件中。
    
    :param chat_data: 要保存的聊天数据，格式为字典
    :param data_file: 存储聊天数据的 JSON 文件路径
    """
    try:
        if data_file.exists() and data_file.stat().st_size != 0:
            # 如果文件存在且非空，读取现有数据
            with open(data_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            # 如果文件不存在或为空，初始化一个空列表
            existing_data = []

        # 将新的聊天数据追加到现有数据中
        existing_data.append(chat_data)

        # 保存更新后的数据到文件
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    except json.JSONDecodeError as e:
        print(f"Failed to load JSON data: {e}")
        # 处理文件内容损坏的情况
        existing_data = []
        existing_data.append(chat_data)
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Failed to save chat data: {e}")
        raise

def load_chat_data(data_file: Path):
    """
    从指定的 JSON 文件中加载聊天数据。
    
    :param data_file: 存储聊天数据的 JSON 文件路径
    :return: 聊天数据的列表
    """
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
    else:
        chat_data = []
    return chat_data

def delete_chat_data(user_id: str, data_file: Path):
    """
    删除指定用户的聊天数据。
    
    :param user_id: 要删除的用户 ID
    :param data_file: 存储聊天数据的 JSON 文件路径
    """
    try:
        if data_file.exists() and data_file.stat().st_size != 0:
            with open(data_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # 过滤掉指定用户的数据
            existing_data = [entry for entry in existing_data if entry['user_id'] != user_id]

            # 保存更新后的数据到文件
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    except json.JSONDecodeError as e:
        print(f"Failed to load JSON data: {e}")
    
    except Exception as e:
        print(f"Failed to delete chat data: {e}")
        raise

def merge_chat_data(data_file: Path, merge_file: Path):
    """
    将现有聊天数据与另一个文件中的聊天数据合并。
    
    :param data_file: 现有聊天数据的 JSON 文件路径
    :param merge_file: 要合并的聊天数据文件路径
    """
    try:
        # 加载现有数据
        existing_data = load_chat_data(data_file)
        # 加载要合并的数据
        merge_data = load_chat_data(merge_file)

        # 合并数据
        combined_data = existing_data + merge_data

        # 保存合并后的数据
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Failed to merge chat data: {e}")
        raise
