from pydantic import BaseModel
from typing import List, Dict

class Config(BaseModel):
    # 关键词列表，用于判断是否与当前话题相关
    relevant_keywords: List[str] = ["机器人", "AI", "人工智能", "机器学习"]

    # 响应概率相关配置
    initial_response_probability: float = 0.05  # 初始响应概率
    max_response_probability: float = 0.95  # 最高响应概率
    response_probability_growth_rate: float = 0.1  # 随时间和对话量增加的概率增速

    # 对话退出条件
    exit_keywords: List[str] = ["再见", "拜拜", "下次再聊", "雪豹闭嘴"]  # 用于识别退出话题的关键词
    max_inactivity_duration: int = 300  # 用户不发言超过此秒数（5分钟）则视为退出话题

    # 其他相关配置
    max_response_length: int = 100  # 生成的回复的最大长度
    min_response_length: int = 1  # 生成的回复的最小长度
