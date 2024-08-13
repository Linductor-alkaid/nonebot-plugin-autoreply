import json
import random
from nonebot import on_message
from nonebot.adapters.onebot.v11 import Message, GroupMessageEvent, PrivateMessageEvent
from nonebot.plugin import PluginMetadata
from pathlib import Path
from datetime import datetime

from .utils.data_handler import save_chat_data
from .data.model.trainer import train_model, load_model

# 定义插件的元数据
__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-autoreply",
    description="记录所有聊天数据并动态训练模型以模拟在QQ中的发言时机和内容的插件",
    usage="安装插件并加载后，自动开始记录数据和训练模型",
    extra={
        "author": "Linductor-alkaid",
        "email": "202200171251@mail.sdu.edu.cn",
        "version": "0.1.0"
    }
)

# 定义数据存储路径
data_file = Path(__file__).parent / "data" / "chat_data.json"
data_file.parent.mkdir(parents=True, exist_ok=True)

# 定义 generate_response 函数
def generate_response(model, user_id):
    """
    根据训练后的模型生成回复。

    :param model: 训练后的模型数据（通常是一个字典）
    :param user_id: 用户ID，用于根据用户历史生成回复
    :return: 生成的回复内容
    """
    # 检查用户是否在模型中
    if user_id in model:
        message_count = model[user_id]['message_count']
        
        # 基于用户的消息计数生成一个简单的回复
        responses = [
            "你已经发送了 {} 条消息，我知道你在想什么！".format(message_count),
            "你又来了，这是第 {} 次发言了！".format(message_count),
            "看来你很喜欢聊天，这是你第 {} 次发言。".format(message_count)
        ]
        return random.choice(responses)
    else:
        return "你好！我还是第一次和你聊天。"

# 消息监听器
chat_listener = on_message()

@chat_listener.handle()
async def handle_chat(event: GroupMessageEvent | PrivateMessageEvent):
    user_id = event.user_id
    group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
    message = event.get_plaintext()
    timestamp = datetime.now().isoformat()

    # 保存聊天数据
    chat_data = {
        "user_id": user_id,
        "group_id": group_id,
        "message": message,
        "timestamp": timestamp,
    }
    save_chat_data(chat_data, data_file)

    # 动态训练模型
    train_model(data_file)

    # 可选：根据训练后的模型生成回复
    # model = load_model()
    # response = generate_response(model, user_id)
    # await chat_listener.finish(response)
