"""Patch 60: both GigaChat 3.5 choices work before the registries load."""

import argparse
import sys
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs


class TestGigaChat35CLI(unittest.TestCase):
    def test_choices_before_detector_imports(self):
        registries = (
            "sglang.srt.function_call.function_call_parser",
            "sglang.srt.parser.reasoning_parser",
        )
        with patch.dict(sys.modules):
            for module in registries:
                sys.modules.pop(module, None)
            parser = argparse.ArgumentParser(exit_on_error=False)
            ServerArgs.add_cli_args(parser)
            for flag in ("--tool-call-parser", "--reasoning-parser"):
                with self.subTest(flag=flag):
                    args = parser.parse_args(["--model-path", "unused", flag, "gigachat35"])
                    self.assertEqual(getattr(args, flag[2:].replace("-", "_")), "gigachat35")
            for module in registries:
                self.assertNotIn(module, sys.modules)
