"""
此模块用于处理与模型相关的操作，包括模型的存储和加载。
"""

from pathlib import Path

# 定义模型文件路径
model_file = Path(__file__).parent / "chat_model.pkl"

# 检查模型文件是否存在
if not model_file.exists():
    # 如果模型文件不存在，可以在这里初始化一个空模型或默认模型
    # 例如，初始化一个空字典并保存
    import pickle
    with open(model_file, 'wb') as f:
        pickle.dump({}, f)
