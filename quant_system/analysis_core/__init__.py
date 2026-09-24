"""
analysis_core — AI 深度投研分析系统（短线引擎/中长线引擎/推演/决策/验证）

本包为 V12.3.0 的现行分析代码载体（2026-08-14 由 analysis_v11 改名而来）。
**目录名与系统版本解耦**：后续版本升级不再改名，只更新
quant_system/__init__.py 的 __version__ 与 CHANGELOG.md（见 VERSIONING.md）。

历史设计文档：项目文档/量化交易系统/AI投研分析系统V11_深度设计方案.md
（文档名为历史遗留，内容仍为现行设计基线；新设计变更记入 CHANGELOG.md）

本包刻意不依赖 ic_factors / backtest 因子路线，全部围绕
"盘面结构 → 情景推演 → 决策预案 → 验证校准"。
"""
