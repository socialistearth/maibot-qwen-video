from maibot_qwen_bilibili.tests.guard_helpers import guarded_call
from pathlib import Path
import asyncio
import importlib.util
import sys
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[1]
NAME = "regression_" + ROOT.name.replace("-", "_")
spec = importlib.util.spec_from_file_location(NAME, ROOT / "plugin.py", submodule_search_locations=[str(ROOT)])
module = importlib.util.module_from_spec(spec)
sys.modules[NAME] = module
spec.loader.exec_module(module)

def test_processed_label_does_not_hide_raw_short_link():
    async def run():
        plugin = module.QwenBilibiliPlugin()
        cfg = plugin.build_default_config()
        cfg["access"]["user_whitelist"] = ["u"]
        plugin.set_plugin_config(cfg)
        plugin._set_context(MagicMock())
        await plugin.handle_incoming_message(message={
            "session_id": "s", "processed_plain_text": "视频分享",
            "raw_message": [{"type": "text", "data": "https://b23.tv/VqztWoh"}],
            "message_info": {"user_info": {"user_id": "u"}},
        })
        plugin._process_link = AsyncMock(return_value="video content")
        result = await guarded_call(plugin, "bilibili_video_summarize", url="https://b23.tv/VqztWoh", stream_id="s", user_id="u")
        assert result["success"] is True
        assert result["stop_after_execution"] is False
        plugin._process_link.assert_awaited_once_with("https://b23.tv/VqztWoh")
    asyncio.run(run())
