import json
import time
import torch
import numpy as np
from datetime import datetime
import asyncio
import random
from pathlib import Path


from nonebot import on_message
from nonebot.adapters.onebot.v11 import Message, GroupMessageEvent, PrivateMessageEvent
from nonebot.adapters.onebot.v12 import MessageSegment
from nonebot.plugin import PluginMetadata
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logging.basicConfig(level=logging.INFO)

from .utils.data_handler import save_chat_data
from .data.model.trainer import train_model
from .utils.model_handler import load_gpt_model

# 配置关键词和超时时间
keywords = ["robot", "bot", "auto", "自动", "雪豹", "io", "机器人", "rng"]
exit_timeout = 300  # 5分钟

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

# 初始化 GPT 模型和分词器并将它们加载到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_gpt_model()
model.to(device)

def generate_response(model, tokenizer, user_data):
    """
    使用 GPT 模型根据用户发言历史生成回复。

    :param model: 训练后的 GPT 模型
    :param tokenizer: GPT 模型的分词器
    :param user_data: 用户的发言频率和时机数据
    :return: 生成的回复内容
    """
    last_message = user_data['last_message']
    if isinstance(last_message, list):  # 如果上次消息是列表，拼接文本
        input_text = " ".join([seg['data'] for seg in last_message if seg['type'] == 'text'])
    else:
        input_text = last_message

    input_text = f"User said: {input_text} " \
                 f"Time since last message: {user_data['time_diff']} seconds."

    # 设置 pad_token 为 eos_token（结束标记）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True)
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()
    outputs = model.generate(
        inputs.to(device), 
        attention_mask=attention_mask.to(device), 
        max_length=100, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 只保留实际的回复内容，不包括上下文信息
    response_text = response_text.replace("User said:", "").strip()
    response_text = response_text.split("Time since last message:")[0].strip()
    
    # 将回复内容区分为文字和图片段
    response_segments = []
    if "image" in response_text:  # 假设 GPT 模型使用 [IMAGE] 代表图片
        response_segments.append({"type": "image", "data": response_text.split("image:")[0].strip()})
    else:
        response_segments.append({"type": "text", "data": response_text})

    return response_segments


def should_respond(user_data: dict) -> bool:
    """
    根据当前对话状态判断机器人是否应该响应。

    :param user_data: 当前用户的对话数据
    :return: 是否响应
    """
    last_message_text = ' '.join([seg['data'] for seg in user_data['last_message'] if seg['type'] == 'text'])
    
    # 检查是否与话题相关
    is_relevant = any(keyword in last_message_text for keyword in keywords)
    
    # 根据相关性设置基础概率
    base_probability = 0.05  # 无关话题时的初始概率
    if is_relevant:
        base_probability = 0.5  # 相关话题时的初始概率
    
    # 使用 sigmoid 函数调整概率
    sigmoid = lambda x: 1 / (1 + torch.exp(-x))
    message_length = sum(len(seg['data']) for seg in user_data['last_message'] if seg['type'] == 'text') 
    probability_input = torch.tensor(message_length / 100.0 + user_data['time_diff'] / 60.0)
    probability = sigmoid(probability_input)
    
    # 最终响应概率
    final_probability = min(base_probability + probability.item(), 1.0)

    # 使用随机数决定是否响应
    return random.random() < final_probability



# 消息监听器

chat_listener = on_message()

@chat_listener.handle()
async def handle_chat(event: GroupMessageEvent | PrivateMessageEvent):
    # 将消息处理任务交给独立的任务处理函数
    asyncio.create_task(process_message(event))

async def process_message(event: GroupMessageEvent | PrivateMessageEvent):
    user_id = event.user_id
    group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
    message = event.message
    timestamp = datetime.now().isoformat()

    # 初始化要记录的消息内容
    recorded_message = []

    # 处理不同类型的消息段
    for seg in message:
        if seg.type == "text":
            recorded_message.append({"type": "text", "data": seg.data["text"]})
        elif seg.type == "image":
            recorded_message.append({"type": "image", "data": seg.data["url"]})

    # 计算与上次发言的时间差
    current_time = time.time()
    time_diff = 0
    if user_id in user_data and 'last_time' in user_data[user_id]:
        time_diff = current_time - user_data[user_id]['last_time']
    
        # 确保 `user_data` 包含所需的键
    if user_id not in user_data:
        user_data[user_id] = {}
    if 'last_message' not in user_data[user_id]:
        user_data[user_id]['last_message'] = []
    if 'last_time' not in user_data[user_id]:
        user_data[user_id]['last_time'] = current_time
    if 'time_diff' not in user_data[user_id]:
        user_data[user_id]['time_diff'] = time_diff
    if 'in_conversation' not in user_data[user_id]:
        user_data[user_id]['in_conversation'] = False

    # 保存聊天数据和时间差
    chat_data = {
        "user_id": user_id,
        "group_id": group_id,
        "message": recorded_message,
        "timestamp": timestamp,
        "time_diff": time_diff
    }
    save_chat_data(chat_data, data_file)

    # 更新用户数据
    user_data[user_id] = {
        "last_message": recorded_message,
        "last_time": current_time,
        "time_diff": time_diff,
        "in_conversation": user_data.get(user_id, {}).get("in_conversation", False)
    }

    # 检查是否包含关键词
    in_conversation = any(keyword in seg['data'] for seg in recorded_message if seg['type'] == 'text' for keyword in keywords)
    if in_conversation:
        user_data[user_id]["in_conversation"] = True  # 进入话题状态

    # 如果没有 `last_message` 键，初始化为空列表
    if "last_message" not in user_data[user_id]:
        user_data[user_id]["last_message"] = []

    # 持久化 user_data
    with open(user_data_file, 'w') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=4)

    # 动态训练模型
    train_model(data_file)

    # 响应概率计算
    if user_data[user_id]["in_conversation"]:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        message_length = sum(len(seg['data']) for seg in user_data['last_message'] if seg['type'] == 'text')
        probability = sigmoid(message_length / 100.0 + user_data['time_diff'] / 60.0)
    else:
        probability = 0.05  # 无关话题时较低的参与概率

    if random.random() > probability:
        return  # 放弃响应

    # 生成回复
    logging.info("开始生成回复...")
    response_segments = generate_response(model, tokenizer, user_data[user_id])
    logging.info(f"生成的回复: {response_segments}")

    # 模拟用户的发言间隔
    delay = time_diff + random.uniform(-1, 1)  # 加入随机延迟
    await asyncio.sleep(delay)

    # 处理生成的回复，根据消息类型发送
    for seg in response_segments:
        if seg['type'] == 'text':
            await chat_listener.send(MessageSegment.text(seg['data']))
        elif seg['type'] == 'image':
            await chat_listener.send(MessageSegment.image(seg['data']))

    # 处理话题退出逻辑
    if user_data[user_id]["in_conversation"] and time_diff > exit_timeout:
        user_data[user_id]["in_conversation"] = False  # 退出话题
        logging.info(f"用户 {user_id} 退出话题。")

    # 保存机器人发送的消息到数据集
    bot_chat_data = {
        "user_id": "3812199284",  # 机器人ID
        "group_id": group_id,
        "message": response_segments,
        "timestamp": datetime.now().isoformat(),
        "time_diff": 0  # 机器人没有上次发言时间
    }
    save_chat_data(bot_chat_data, data_file)