from nonebot import on_message
from nonebot.adapters.onebot.v11 import MessageEvent
from datetime import datetime
import csv
import pickle
from collections import Counter

# 定义消息处理器
record_message = on_message()

# 尝试加载已存在的频率模型
try:
    with open('models/frequency_model.pkl', 'rb') as f:
        frequency_model = pickle.load(f)
except FileNotFoundError:
    # 如果模型文件不存在，初始化一个空的 Counter 对象
    frequency_model = Counter()

@record_message.handle()
async def handle_message(event: MessageEvent):
    user_id = event.get_user_id()
    message = event.get_plaintext()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存到 CSV 文件
    with open('message_records.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, message, timestamp])

    # 更新频率模型
    frequency_model.update([message])

    # 定期保存模型，例如每处理 100 条消息保存一次
    if sum(frequency_model.values()) % 100 == 0:
        with open('models/frequency_model.pkl', 'wb') as f:
            pickle.dump(frequency_model, f)

    await record_message.finish()

