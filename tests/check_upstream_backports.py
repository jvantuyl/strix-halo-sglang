"""CPU integration checks for patch 59; run in the patched image."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from sglang.srt.managers.scheduler import Scheduler


class TestMixedCapacity(unittest.TestCase):
    def test_capacity_after_filter_preserves_decoder_for_next_round(self):
        # Execute the actual final admission block. The harness starts after the
        # prefill has allocated its slots; no model, GPU or policy simulation.
        method = ast.parse(textwrap.dedent(inspect.getsource(Scheduler._get_new_batch_prefill_raw))).body[0]
        branch, ret = method.body[-2:]
        self.assertIsInstance(branch, ast.If)
        self.assertIsInstance(ret, ast.Return)
        method.body = [branch, ret]
        method.args = ast.parse("def f(self, allow_mixed, new_batch, running_batch): pass").body[0].args
        method.returns = None
        namespace = {"ScheduleBatch": NS}
        exec(compile(ast.Module([method], []), "<mixed admission>", "exec"), namespace)
        dispatch = namespace[method.name]
        scheduler = NS(enable_overlap=True)
        pre = Mock(return_logprob=False, input_embeds=None, decoding_reqs=None)
        rows = [NS(beam_group=None)]
        running = Mock(reqs=rows, return_logprob=False)
        running.is_empty.return_value = False
        running.check_decode_mem.side_effect = [False, True]
        self.assertEqual(dispatch(scheduler, True, pre, running), (pre, running))
        running.prepare_for_decode.assert_not_called()
        pre.mix_with_running.assert_not_called()
        self.assertIsNone(pre.decoding_reqs)
        self.assertIs(running.reqs, rows)
        later = Mock(return_logprob=False, input_embeds=None)
        mixed, remainder = dispatch(scheduler, True, later, running)
        self.assertIs(mixed.decoding_reqs, rows)
        self.assertEqual(remainder.reqs, [])
        self.assertEqual([m[0] for m in running.method_calls],
                         ["is_empty", "filter_batch", "is_empty", "check_decode_mem"] * 2
                         + ["prepare_for_decode"])
        later.mix_with_running.assert_called_once_with(running)


if __name__ == "__main__":
    unittest.main()
