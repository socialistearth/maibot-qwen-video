# Qwen B站视频理解：MaiBot 1.2.4 适配版

> 本文仅记录 1.2.0 的历史验证，不代表 1.4.0 的兼容范围。当前版本最低要求 MaiBot 1.3.5 / SDK 2.10.0；请阅读 [当前升级说明](UPGRADE_1.4.0.md)。

插件版本：1.2.0。目标：MaiBot **1.2.4** / **maibot-plugin-sdk >=2.8.0,<3**。
宿主版本范围锁定为 1.2.4，未声明其他版本的兼容性。配置结构未变，因此原 config_version 可以保留。

## 本次修改

同时读取当前入站消息的预处理文本与原始文本段；保留短链授权、白名单和 Qwen 完整视频处理，成功返回后继续 Planner 回复。

## 升级方法

1. 先停止 MaiBot 容器，备份现有插件目录及插件持久化数据。
2. 将本包解压到实际的插件父目录，覆盖原 `maibot_qwen_bilibili/` 文件夹；不要另建第二份插件目录。
3. 保留原 `config.toml` 和整个插件数据目录，不要删除再重装。本包不包含运行时 config.toml，不会覆盖 API Key、Cookie、白名单或自定义提示词。
4. 启动 MaiBot，在 WebUI 确认插件版本为 1.2.0，检查启用状态及配置。

若你的容器仍名为 maim-bot-core：

```bash
docker stop maim-bot-core
# 将 /实际插件父目录 换为你部署时使用的宿主机目录。
unzip -o maibot_qwen_bilibili_v1.2.0-MaiBot1.2.4.zip -d /实际插件父目录
docker start maim-bot-core
```

不修改 Compose、适配器或 MaiBot 主程序。新安装由 SDK 配置模型生成默认配置；便利贴/定时搜索还保留 config.default.toml。
升级后首次测试 B站工具请在当前聊天重新发送链接：入站授权仅存在内存中，重启不会保留。

## 验证范围

基于 MaiBot 标签 1.2.4、提交 `21cd74d81d47b6f77ba5ab7116a88916e957d15b`，使用真实 SDK 2.8.0。
四包回归测试合计 **49 项通过**（包含原有测试及本次新增测试）。
用该版本实际 PluginLoader/ManifestValidator 加载四个插件，并验证 SDK 配置 Schema 与组件注册。
用实际 Context Item 反序列化函数校验便利贴注入；用模拟 RPC/视频分析验证 Steam 成功与失败分支、便利贴生命周期/工具结果、B站原始短链授权。
验证环境仅替代宿主日志与版本定位等启动设施，没有运行完整 MaiBot/QQ/Docker 服务，未调用 Steam、B站或付费 Qwen 实网接口；不等同于服务器端实测。

源码基准：https://github.com/Mai-with-u/MaiBot/tree/1.2.4

测试命令（SDK 2.8.0、pytest、aiohttp、httpx、Pillow、socksio 可用）：

```bash
# 在插件父目录执行；B站包需要父目录在 PYTHONPATH 中。
PYTHONPATH=. python -m pytest -q maibot_qwen_bilibili/tests --import-mode=importlib
```

具体配置与功能见 README.md。
