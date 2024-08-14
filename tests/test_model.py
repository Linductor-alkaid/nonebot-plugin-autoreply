import unittest
from pathlib import Path
from utils.model_handler import load_model, save_model, update_model

class TestModelHandler(unittest.TestCase):

    def setUp(self):
        """在每个测试前运行，设置测试环境。"""
        # 定义测试模型文件路径
        self.test_model_file = Path("test_chat_model.pkl")
        # 初始化测试数据
        self.test_data = {
            "user_id": "2052046346",
            "message": "这是一个测试消息"
        }
        # 初始化模型
        self.model = {}

    def tearDown(self):
        """在每个测试后运行，清理测试环境。"""
        # 删除测试模型文件
        if self.test_model_file.exists():
            self.test_model_file.unlink()

    def test_save_and_load_model(self):
        """测试模型的保存和加载功能。"""
        # 保存模型
        save_model(self.model, self.test_model_file)
        # 加载模型
        loaded_model = load_model(self.test_model_file)
        # 断言模型加载正确
        self.assertEqual(loaded_model, self.model)

    def test_update_model(self):
        """测试模型的更新功能。"""
        # 更新模型
        updated_model = update_model(self.test_data, self.model)
        # 断言模型更新正确
        self.assertIn("123456", updated_model)
        self.assertEqual(updated_model["123456"]["message_count"], 1)

        # 再次更新模型
        updated_model = update_model(self.test_data, updated_model)
        # 断言模型消息计数增加
        self.assertEqual(updated_model["123456"]["message_count"], 2)

if __name__ == "__main__":
    unittest.main()
