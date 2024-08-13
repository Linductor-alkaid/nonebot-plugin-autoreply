# nonebot-plugin-autoreply

**nonebot-plugin-autoreply** 是一个用于记录所有聊天数据并动态训练模型，以模拟人在QQ中的发言时机和内容的 NoneBot2 插件。

## 功能简介

- 记录群聊和私聊消息，包括用户ID、群组ID、消息内容和时间戳。
- 基于记录的聊天数据动态训练模型，模拟QQ聊天中的人类发言时机和内容。
- 模型自动更新，不断提升回复的准确性。

## 安装

1. 克隆项目代码到你的本地环境：

   ```bash
   git clone https://github.com/Linductor-alkaid/nonebot-plugin-autoreply.git
   ```

2. 安装依赖：

   在项目根目录下，创建并激活虚拟环境（可选）：

   ```bash
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
   ```

3. 将插件加载到 NoneBot2 项目中：

   在你的 NoneBot2 项目的 `bot.py` 中添加：

   ```python
   nonebot.load_plugin('nonebot_plugin_autoreply')
   ```

## 使用方法

- 插件加载后将自动开始记录所有收到的消息，并基于这些数据进行模型训练。
- 你可以通过自定义 `config.py` 文件来调整插件的配置，例如数据存储路径和模型训练参数。

## 文件结构

```
nonebot-plugin-autoreply/
│
├── __init__.py              # 插件初始化文件
├── plugin.py                # 插件的主要逻辑
├── config.py                # 插件的配置文件（可选）
├── data/                    # 数据存储目录
│   ├── chat_data.json       # 存储聊天数据的 JSON 文件
│   └── model/               # 模型存储目录
│       ├── chat_model.pkl   # 训练好的模型文件
│       └── trainer.py       # 模型训练与更新逻辑
├── utils/                   # 工具模块
│   ├── data_handler.py      # 数据处理相关工具函数
│   └── model_handler.py     # 模型处理相关工具函数
└── tests/                   # 测试代码
    ├── test_plugin.py       # 插件功能测试
    └── test_model.py        # 模型功能测试
```

## 贡献

如果你有兴趣为此项目做出贡献，欢迎提交Pull Request或提出Issue。

## 作者

- GitHub: [Linductor-alkaid](https://github.com/Linductor-alkaid)
- Email: 202200171251@mail.sdu.edu.cn

