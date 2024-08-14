import unittest
from pathlib import Path
from nonebot.adapters.onebot.v11 import GroupMessageEvent, PrivateMessageEvent, Message
from nonebot_plugin_autoreply.plugin import handle_chat
from utils.data_handler import load_chat_data

class TestPlugin(unittest.TestCase):

    def setUp(self):
        """在每个测试前运行，设置测试环境。"""
        # 定义测试数据文件路径
        self.test_data_file = Path("test_chat_data.json")
        # 初始化测试环境，确保数据文件为空
        self.test_data_file.write_text("[]", encoding='utf-8')

    def tearDown(self):
        """在每个测试后运行，清理测试环境。"""
        # 删除测试数据文件
        if self.test_data_file.exists():
            self.test_data_file.unlink()

    async def test_handle_group_message(self):
        """测试群聊消息处理。"""
        # 构造一个模拟的群聊消息事件
        event = GroupMessageEvent(
            user_id=123456,
            group_id=654321,
            message=Message("这是一个群聊测试消息"),
            raw_message="这是一个群聊测试消息"
        )

        # 处理消息事件
        await handle_chat(event)

        # 加载保存的聊天数据
        chat_data = load_chat_data(self.test_data_file)

        # 断言数据保存正确
        self.assertEqual(len(chat_data), 1)
        self.assertEqual(chat_data[0]['user_id'], 2052046346)
        self.assertEqual(chat_data[0]['group_id'], 118393524)
        self.assertEqual(chat_data[0]['message'], "这是一个群聊测试消息")

    async def test_handle_private_message(self):
        """测试私聊消息处理。"""
        # 构造一个模拟的私聊消息事件
        event = PrivateMessageEvent(
            user_id=789012,
            message=Message("这是一个私聊测试消息"),
            raw_message="这是一个私聊测试消息"
        )

        # 处理消息事件
        await handle_chat(event)

        # 加载保存的聊天数据
        chat_data = load_chat_data(self.test_data_file)

        # 断言数据保存正确
        self.assertEqual(len(chat_data), 1)
        self.assertEqual(chat_data[0]['user_id'], 2052046346)
        self.assertIsNone(chat_data[0]['group_id'])
        self.assertEqual(chat_data[0]['message'], "这是一个私聊测试消息")

if __name__ == "__main__":
    unittest.main()
