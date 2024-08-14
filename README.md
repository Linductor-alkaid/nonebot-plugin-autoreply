# nonebot-plugin-autoreply

**nonebot-plugin-autoreply** 是一个用于记录所有聊天数据并动态训练生成式预训练模型（GPT）以模拟人在QQ中的发言时机和内容的 NoneBot2 插件。

## 功能简介

- 记录群聊和私聊消息，包括用户ID、群组ID、消息内容和时间戳。
- 基于记录的聊天数据动态训练生成式预训练模型（GPT），模拟QQ聊天中的人类发言时机和内容。
- 模型自动更新，不断提升回复的准确性，并保存用户的发言频率和时机数据。
- 持久化用户数据，以便在 bot 重启后仍能保留用户的发言频率和时机信息。

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

   然后安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 安装 PyTorch（或 TensorFlow），以支持 GPT 模型：

   例如，安装 PyTorch：

   ```bash
   pip install torch
   ```

   或者安装 TensorFlow：

   ```bash
   pip install tensorflow
   ```
4. 安装本地模型库：

   执行./data/model/getmodel.py

   ```bash
   python3 getmodel.py
   ```

5. 将插件加载到 NoneBot2 项目中：

   在你的 NoneBot2 项目的 `bot.py` 中添加：

   ```python
   nonebot.load_plugin('nonebot_plugin_autoreply')
   ```

## 使用方法

- 插件加载后将自动开始记录所有收到的消息，并基于这些数据进行模型训练。
- 你可以通过自定义 `config.py` 文件来调整插件的配置，例如数据存储路径和模型训练参数。

## 更新日志

### v0.1.1
- 新增使用生成式预训练模型（GPT）来生成回复，模拟用户发言时机和风格。
- 添加了用户数据的持久化功能，以便在 bot 重启后保留用户的发言历史。
### v0.1.2
- 新增区分接收文本信息和图像信息
### v0.1.3
- 新增预训练文本的接口：文件夹data/pretrain中放入txt文件即可
### v0.1.4
- 修复了部分bug
### v0.1.5
- 更新了使用显卡驱动训练模型，同时修复了已知的大部分bug

## 文件结构

```
nonebot-plugin-autoreply/
│
├── nonebot_plugin_autoreply/   # 插件代码目录
│   ├── __init__.py
│   ├── plugin.py
│   ├── config.py
│   ├── data/
│   │   ├── chat_data.json       # 存储聊天数据的 JSON 文件
│   │   ├── user_data.json       # 持久化用户发言频率和时机数据
│   │   ├── pretrain/            # 预训练文本接口文件夹（在该文件夹中放入txt文件即可预训练）
│   │   └── model/               # 模型存储目录
│   │       ├── getmodel.py      # 获取训练模型
│   │       ├── chat_model.pkl   # 训练好的 GPT 模型文件
│   │       └── trainer.py       # 模型训练与更新逻辑
│   ├── utils/                   # 工具模块
│   │   ├── data_handler.py      # 数据处理相关工具函数
│   │   └── model_handler.py     # 模型处理相关工具函数
│   └── tests/                   # 测试代码
│       ├── test_plugin.py       # 插件功能测试
│       └── test_model.py        # 模型功能测试
├── README.md                    # 项目说明文件
├── requirements.txt             # 依赖包列表
└── LICENSE                      # 许可证文件（可选）
```

## 贡献

如果你有兴趣为此项目做出贡献，欢迎提交Pull Request或提出Issue。

## 作者

- GitHub: [Linductor-alkaid](https://github.com/Linductor-alkaid)
- Email: 202200171251@mail.sdu.edu.cn
