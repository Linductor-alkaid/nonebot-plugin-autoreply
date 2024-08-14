from pathlib import Path

import nonebot
from nonebot import get_plugin_config
from nonebot.plugin import PluginMetadata

from .config import Config

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-autoreply",
    description="记录所有聊天数据并动态训练模型以模拟在QQ中的发言时机和内容的插件",
    usage="安装插件并加载后，自动开始记录数据和训练模型",
    config=Config,
    extra={
        "author": "Linductor-alkaid",
        "email": "202200171251@mail.sdu.edu.cn",
        "version": "0.1.0"
    }
)

from .plugin import *

config = get_plugin_config(Config)

sub_plugins = nonebot.load_plugins(
    str(Path(__file__).parent.joinpath("plugins").resolve())
)

