"""Regression for post-Hook declared-parameter filtering of video tools."""
from __future__ import annotations

import unittest

from maibot_qwen_bilibili.plugin import QwenBilibiliPlugin
from maibot_qwen_bilibili.tool_guard import IDENTITY_FIELDS, TICKET_FIELD, VideoToolGuard


def declared_arguments(arguments, schema):
    """Model Core's declared-property filter for these flat object schemas.

    Keep this fixture independent of a full Core checkout and model/API access.
    A field absent from properties must be removed even if a Hook supplied it.
    """
    return {key: value for key, value in arguments.items() if key in schema["properties"]}


class Core145CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.filter = declared_arguments
        self.schemas = {
            c['name']: c['metadata']['parameters_raw']
            for c in QwenBilibiliPlugin().get_components() if c['type'] == 'TOOL'
        }

    def issue(self, tool, argument, value):
        guard = VideoToolGuard()
        stamped = guard.stamp([{
            'item_type': 'FunctionCallItem',
            'tool_call': {'func_name': tool, 'args': {argument: value}},
        }], 'trusted-session', 1)
        return guard, stamped[0]['tool_call']['args']

    def test_bilibili_ticket_survives_core_and_remains_single_use(self):
        tool, key, value = 'bilibili_video_summarize', 'url', 'https://b23.tv/test'
        guard, args = self.issue(tool, key, value)
        forwarded = self.filter(args, self.schemas[tool])
        self.assertEqual(guard.consume(forwarded.get(TICKET_FIELD), tool, value), 'trusted-session')
        self.assertIsNone(guard.consume(forwarded.get(TICKET_FIELD), tool, value))

    def test_qq_ticket_survives_core_and_remains_single_use(self):
        tool, key, value = 'qq_video_summarize', 'message_id', '123'
        guard, args = self.issue(tool, key, value)
        forwarded = self.filter(args, self.schemas[tool])
        self.assertEqual(guard.consume(forwarded.get(TICKET_FIELD), tool, value), 'trusted-session')
        self.assertIsNone(guard.consume(forwarded.get(TICKET_FIELD), tool, value))

    def test_argument_tampering_still_rejected_after_core_filter(self):
        for tool, key in [('bilibili_video_summarize', 'url'), ('qq_video_summarize', 'message_id')]:
            with self.subTest(tool=tool):
                guard, args = self.issue(tool, key, 'original')
                forwarded = self.filter(args, self.schemas[tool])
                self.assertIsNone(guard.consume(forwarded.get(TICKET_FIELD), tool, 'changed'))

    def test_declaring_ticket_does_not_make_model_ticket_or_identity_trusted(self):
        for tool, key in [('bilibili_video_summarize', 'url'), ('qq_video_summarize', 'message_id')]:
            with self.subTest(tool=tool):
                guard = VideoToolGuard()
                args = {key: 'original', TICKET_FIELD: 'model-forged',
                        **dict.fromkeys(IDENTITY_FIELDS, 'victim')}
                forwarded = self.filter(args, self.schemas[tool])
                self.assertFalse(set(IDENTITY_FIELDS) & forwarded.keys())
                self.assertIsNone(guard.consume(forwarded.get(TICKET_FIELD), tool, 'original'))
