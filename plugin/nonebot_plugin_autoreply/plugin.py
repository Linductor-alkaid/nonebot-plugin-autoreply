import json
import time
from datetime import datetime
from pathlib import Path

from nonebot import on_message
from nonebot.adapters.onebot.v11 import Message, GroupMessageEvent, PrivateMessageEvent
from nonebot.plugin import PluginMetadata
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .utils.data_handler import save_chat_data
from .data.model.trainer import train_model, load_model
from .utils.model_handler import load_gpt_model, save_gpt_model


# 定义 user_data 文件的路径
user_data_file = Path(__file__).parent / "data" / "user_data.json"

# 加载 user_data，如果文件不存在则初始化为空字典
if user_data_file.exists():
    with open(user_data_file, 'r') as f:
        user_data = json.load(f)
else:
    user_data = {}


# 定义插件的元数据
__plugin_meta__ = PluginMetadata(
    name="AutoReply",
    description="记录所有聊天数据并动态训练GPT模型以模拟在QQ中的发言时机和内容的插件",
    usage="安装插件并加载后，自动开始记录数据和训练模型",
)

# 定义数据存储路径
data_file = Path(__file__).parent / "data" / "chat_data.json"
data_file.parent.mkdir(parents=True, exist_ok=True)

# 初始化 GPT 模型和分词器
model, tokenizer = load_gpt_model()  # 此函数从 model_handler.py 中加载模型

# 用户数据，用于存储用户发言频率和时机信息
user_data = {}

def generate_response(model, tokenizer, user_data):
    """
    使用 GPT 模型根据用户发言历史生成回复。

    :param model: 训练后的 GPT 模型
    :param tokenizer: GPT 模型的分词器
    :param user_data: 用户的发言频率和时机数据
    :return: 生成的回复内容
    """
    # 使用 GPT 模型生成回复
    input_text = f"User said: {user_data['last_message']} " \
                 f"Time since last message: {user_data['time_diff']} seconds."
    
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# 消息监听器
chat_listener = on_message()

@chat_listener.handle()
async def handle_chat(event: GroupMessageEvent | PrivateMessageEvent):
    user_id = event.user_id
    group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
    message = event.get_plaintext()
    timestamp = datetime.now().isoformat()

    # 计算与上次发言的时间差
    current_time = time.time()
    if user_id in user_data and 'last_time' in user_data[user_id]:
        time_diff = current_time - user_data[user_id]['last_time']
    else:
        time_diff = 0

    # 保存聊天数据和时间差
    chat_data = {
        "user_id": user_id,
        "group_id": group_id,
        "message": message,
        "timestamp": timestamp,
        "time_diff": time_diff
    }
    save_chat_data(chat_data, data_file)

    # 更新用户数据
    user_data[user_id] = {
        "last_message": message,
        "last_time": current_time,
        "time_diff": time_diff
    }

    # 持久化 user_data
    with open(user_data_file, 'w') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=4)

    # 动态训练模型
    train_model(data_file)

    # 生成并发送回复
    response = generate_response(model, tokenizer, user_data[user_id])
    await chat_listener.finish(response)
