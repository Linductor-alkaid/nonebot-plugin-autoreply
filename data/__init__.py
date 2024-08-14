"""
此模块用于存储聊天数据和训练的模型文件。
"""

# 初始化数据目录结构
import os
from pathlib import Path

# 定义数据目录
data_dir = Path(__file__).parent
model_dir = data_dir / "model"

# 确保目录存在
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
