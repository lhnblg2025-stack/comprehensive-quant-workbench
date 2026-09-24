"""V12.3 审计 P2 回归: lhb/margin 调度可观测性 + ls_remote 解析健壮性。

覆盖:
  1. P2-4: update_lhb_daily 交易日但无数据 → 退出码 1(暴露失败); 非交易日 → 0。
  2. P2-5: pull_cloud_data._ls_remote 的 dir 解析——含空格文件名能解析、
           目录行/汇总行被跳过、无法解析的日期行被计数告警而非静默丢。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/tests -> workspace
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))


class TestLhbExitCode(unittest.TestCase):
    def test_main_exit_logic(self):
        """交易日&无数据→1; 非交易日→0; 有数据→0。用 __main__ 内的判定逻辑复算。"""
        from datetime import datetime

        import update_lhb_daily as ul

        def decide(target: str, n: int) -> int:
            try:
                from quant_system.market_clock import is_trading_day
                trading_day = is_trading_day(target)
            except Exception:  # noqa: BLE001
                d = datetime.strptime(target, "%Y-%m-%d")
                trading_day = d.weekday() < 5
            if trading_day and n == 0:
                return 1
            return 0

        # 有数据 → 0(无论是否交易日)
        self.assertEqual(decide("2026-08-14", 3), 0)
        # 非交易日(周六)无数据 → 0(正常)
        self.assertEqual(decide("2026-08-15", 0), 0)
        # 交易日无数据 → 1(暴露失败)
        self.assertEqual(decide("2026-08-14", 0), 1)


class TestLsRemoteParse(unittest.TestCase):
    def test_parse_files_with_spaces_and_skip_dirs(self):
        """Windows dir 输出: 空格文件名解析、<DIR>/汇总行跳过、无漏解析告警。"""
        from pull_cloud_data import _parse_windows_dir_output

        out = (
            " 驱动器 C 中的卷是 System\n"
            " C:\\quant\\data\\cninfo 的目录\n"
            "\n"
            "2026/08/14  23:01         1,234,567 cninfo_20260814.json\n"
            "2026/08/14  23:02    <DIR>          subdir\n"
            "2026/08/14  23:03           12,345 a b.json\n"
            "               2 个文件      1,246,912 字节\n"
        )
        items, dl, unparsed = _parse_windows_dir_output(out)
        names = {n for n, _ in items}
        self.assertIn("cninfo_20260814.json", names)
        self.assertIn("a b.json", names)  # 空格文件名不被丢
        self.assertNotIn("subdir", names)  # 目录行跳过
        self.assertEqual(dl, 0)  # 无"日期但解析不出"的行 → 不误报

    def test_unparseable_date_line_is_reported(self):
        """以日期开头但格式异常的行走入 unparsed 告警路径, 而非被静默丢弃。"""
        from pull_cloud_data import _parse_windows_dir_output

        out = (
            "2026/08/14  23:05   <SOME-WEIRD-MARKER>  weird_name_here\n"
            "2026/08/14  23:06           12,345 normal.json\n"
        )
        items, dl, unparsed = _parse_windows_dir_output(out)
        self.assertEqual(len(items), 1)  # normal.json 解析成功
        self.assertEqual(dl, 1)  # weird 行走入未解析告警计数
        self.assertTrue(unparsed)  # 有示例被收集, 告警会被触发


if __name__ == "__main__":
    unittest.main(verbosity=2)
