from nonebot import on_message
from nonebot.adapters.onebot.v11 import MessageEvent
import random

# 引入实时更新的频率模型
from plugins.record_message import frequency_model

# 定义回复处理器
reply_message = on_message()

@reply_message.handle()
async def handle_reply(event: MessageEvent):
    message = event.get_plaintext()
    
    # 基于实时频率模型生成回复
    response = None
    if frequency_model:
        for msg, _ in frequency_model.most_common():
            if msg in message:
                response = msg
                break

    # 如果没有匹配项，则随机选择常见回复
    if not response:
        response = random.choice([msg for msg, _ in frequency_model.most_common()])

    await reply_message.finish(response)

