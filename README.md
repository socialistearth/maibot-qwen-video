# Qwen B站与QQ视频理解

麦麦可通过本插件调用 Qwen-Omni 分析 Bilibili 链接和 QQ 直接发送的视频，再根据结果自然回复。

**当前版本：1.4.0。聊天身份防护已整合进插件，无需修改 MaiBot 主程序。**

## 功能

- B站长链、BV/av、`b23.tv` 等短链及分P解析，优先使用合并音视频，必要时使用 DASH 视频轨与音频轨。
- QQ 群聊、私聊中直接发送的 MP4 视频，默认选择当前会话最近 15 分钟内收到的视频。
- 用户/群白名单、时长与大小限制、分析缓存、并发控制、临时错误降级重试。
- 两个工具直接对 Planner 可见：`bilibili_video_summarize` 和 `qq_video_summarize`。
- 分析结果交回麦麦组织回复；入站 Hook 只记录授权，不下载视频或调用模型。
- 内置一次性会话授权，拦截伪造聊天身份、跨会话调用、参数篡改和凭据重放。

不支持普通文件附件中的 MP4、合并转发内视频或任意网站视频。
“完整音视频”指提交完整媒体并保留音轨；模型仍使用配置的视觉采样，不保证识别每一帧、每个细节。

## 兼容性

| 项目 | 要求 |
| --- | --- |
| MaiBot | 最低 1.3.5；manifest 上界 1.3.999 |
| 插件 SDK | 最低 2.10.0；manifest 上界 2.999.999 |
| Python | 使用 MaiBot 自带环境；本地离线测试为 Python 3.12 |
| QQ 适配器 | 提供 `adapter.napcat.message.get_msg` v1 公开 API，返回结构化 OneBot 消息；既有部署使用 SnowLuma |
| 额外依赖 | `aiohttp>=3.12.14,<4` |

较宽的上界用于允许后续小版本更新，不代表未来版本已逐一验证。
1.4.0 已完成官方**未经修改的 MaiBot 1.3.5** 参数处理、Context Item 序列化及 **SDK 2.10.0** 离线集成检查。
不需要 `ffmpeg`、`yt-dlp` 或额外开放端口。

## 安装

### WebUI

在插件管理中使用本仓库地址安装：

```text
https://github.com/socialistearth/maibot-qwen-video
```

也可从 [Releases](https://github.com/socialistearth/maibot-qwen-video/releases) 下载源码 ZIP，使用 WebUI 的 ZIP 安装入口。
市场收录后可搜索「Qwen B站与QQ视频理解」。

### 手动安装

在 MaiBot 根目录执行：

```bash
git clone https://github.com/socialistearth/maibot-qwen-video.git plugins/maibot_qwen_bilibili
python -m pip install -r plugins/maibot_qwen_bilibili/requirements.txt
```

使用 Docker 时，文件应放在实际挂载的插件目录；依赖应安装在运行 MaiBot 的 Python 环境中。
随后在 WebUI 重载并启用插件。仓库不包含 `config.toml`，首次加载由 SDK 生成默认配置。

已有同 ID 插件时请按 [升级说明](UPGRADE_1.4.0.md) 更新原目录，不要安装第二份。

## 配置与使用

进入 `WebUI → 插件管理 → Qwen B站与QQ视频理解 → 配置`：

1. 填写百炼 API Key、与密钥地域匹配的 API 地址及可用的音视频模型。
2. 添加允许使用的用户 QQ 或群 QQ。默认白名单为空且 `allow_when_empty=false`，不会调用付费模型。
3. QQ 直发视频保持 `chat_video.enabled=true`，确认适配器支持上述消息查询 API。
4. B站需要登录权限的视频可填写自己的 Cookie。

默认模型名为 `qwen3.5-omni-plus`，默认地址为
`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。以你的百炼控制台实际可用模型和地域为准。
不要使用普通文本模型；仅支持画面的模型不能提供音轨理解。

白名单模式 `any` 为用户或群任一命中；`all` 为当前会话适用的已配置列表均需命中。
私聊不会使用群白名单代替用户授权。QQ 与 B站共用白名单和限制。

在允许的会话发送：

```text
/qwen视频状态
```

随后发送 B站链接或直接发送 QQ 视频，并说“帮我总结一下这个视频”。
Planner 选择对应工具后才执行分析。重载前、超过 15 分钟或下载地址过期的视频应重新发送。

## 防护与数据处理

官方 `maisaka.planner.after_response` Hook 提供真实会话。插件为视频调用附加一次性授权，
工具消费授权后使用绑定的会话核验入站消息和当前白名单。
防护 Hook 缺失、凭据过期或被篡改时拒绝执行，不回退到模型提供的聊天身份。
本插件不读取或改写宿主源码；设计见 [1.4.0 升级说明](UPGRADE_1.4.0.md)。

QQ 视频仅从代码允许的 QQ 公共媒体域名下载，检查解析地址、重定向、大小及 MP4 实际时长。
QQ 媒体链接不返回 Planner；缓存按会话、消息和视频标识隔离。
B站缓存按视频、分P、模型和分析设置区分。默认缓存 168 小时、最多 200 条，命中时避免重复模型调用。

视频媒体、元数据和分析提示词会发送给配置的模型服务。
`auto` 对小媒体使用 Base64，对大媒体使用百炼临时上传；云端留存规则以服务商为准。
QQ 视频始终先下载并检查，`direct_url` 设置也不会绕过检查。
本地临时媒体在任务结束时清理；分析文字保存在 MaiBot 分配的插件数据目录中。
B站 Cookie 不发送给 Qwen。模型调用可能产生费用，请根据使用范围设置白名单与限制。

## 常见问题

- **缺少一次性会话授权**：确认 MaiBot 版本至少 1.3.5，且 `qwen_video_tool_session_guard` Hook 已启用；重载后重新请求。不要单独禁用该 Hook。
- **没有有效入站记录**：在当前会话重新发送视频或链接；仅处理插件实际观察到的近期消息。
- **QQ 下载地址已过期**：重新直接发送视频；旧消息查询不保证刷新下载地址。
- **消息查询 API 不可用**：检查适配器是否注册 `adapter.napcat.message.get_msg` v1；仅运行 OneBot 服务并不等同于已注册插件 API。
- **401 / invalid_api_key**：检查密钥、API 地址与地域。
- **B站风控或权限错误**：检查网络与 Cookie；插件不绕过视频自身的登录、会员或地区权限。
- **媒体过大或时长超限**：检查插件及模型服务限制；无法确认 MP4 时长时拒绝提交。

## 开发与验证

在仓库根目录执行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests --import-mode=importlib
```

1.4.0 发布前通过 79 项离线测试及 53 项子用例，另通过官方未打补丁的 MaiBot 1.3.5 / SDK 2.10.0 集成检查。
覆盖会话伪造、重放、并发授权、QQ 消息身份核验、媒体限制、缓存及 B站原有功能。
本次公开发布验证未调用付费模型，未执行真实 QQ 视频模型端到端测试。

插件使用 MIT 许可证，保留原有 MaiBot Community 版权声明，由
[socialistearth](https://github.com/socialistearth) 维护发布。
问题请提交到 [Issues](https://github.com/socialistearth/maibot-qwen-video/issues)。
