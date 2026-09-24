"""Public quant-web API catalog for the decision workbench.

The catalog is intentionally small and stable. It describes user-facing domains,
not implementation versions, so the UI can discover capabilities without parsing
server.py or guessing between legacy v11/v12 routes.
"""
from __future__ import annotations

from typing import Any


API_CATALOG_CONTRACT = "quant-web.api-catalog.v1"


API_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "decision",
        "label": "统一决策",
        "icon": "compass",
        "primary": True,
        "read_path": "/api/decision_snapshot?mode=after_close",
        "description": "市场状态、主线、资金、风险门控和候选机会的统一快照",
        "freshness": "按数据实际日期展示，日期不一致只保留观察",
    },
    {
        "id": "market",
        "label": "市场总览",
        "icon": "activity",
        "primary": True,
        "read_path": "/api/market_recap",
        "description": "指数、市场宽度、情绪温度、涨停梯队与资金方向",
        "freshness": "实时或最近已确认交易日",
    },
    {
        "id": "stock",
        "label": "个股决策台",
        "icon": "search",
        "primary": True,
        "read_path": "/api/stock_overview?symbol=002714",
        "description": "价格、趋势、个股资金、行业暴露、研报证据和交易门控",
        "freshness": "每个证据源独立显示数据日",
    },
    {
        "id": "research",
        "label": "统一研报",
        "icon": "file-text",
        "primary": True,
        "read_path": "/api/research_report?type=daily_review",
        "description": "research_report.v1 结构化事实、传导、风险、行动与来源",
        "freshness": "报告日期与来源日期分离",
    },
    {
        "id": "factors",
        "label": "因子验证",
        "icon": "layers",
        "primary": False,
        "read_path": "/api/factor_quality",
        "deep_paths": ["/api/factor_scan", "/api/factor_ic", "/api/factor_combine", "/api/factor_library"],
        "description": "质量账本、IC/OOS、权重和回测门禁",
        "freshness": "以质量账本和验证窗口为准",
    },
    {
        "id": "risk",
        "label": "组合风控",
        "icon": "shield",
        "primary": False,
        "read_path": "/api/portfolio",
        "deep_page": "/v4.html",
        "deep_paths": ["/api/v4/portfolio/optimize", "/api/v4/risk/check", "/api/v4/performance/summary", "/api/v4/performance/factor_attribution"],
        "description": "纸面账户、风险暴露、组合优化和压力测试",
        "freshness": "读取当前纸面账户和本地风险计算",
    },
    {
        "id": "strategy",
        "label": "策略回测",
        "icon": "trending-up",
        "primary": False,
        "read_path": "/api/backtest_dashboard",
        "deep_paths": ["/api/backtest", "/api/backtest_almgren", "/api/v8_backtest", "/api/strategy_compare", "/api/event_backtest"],
        "description": "策略扫描、成本后回测、优化、策略对比和事件回测",
        "freshness": "以回测区间、成本假设和产物生成时间为准",
    },
    {
        "id": "ops",
        "label": "数据与运维",
        "icon": "server",
        "primary": False,
        "read_path": "/api/data_health",
        "deep_paths": ["/api/tasks_status", "/api/alerts_status", "/api/warehouse_status", "/api/cache_stats", "/api/decision_chain", "/api/execution_ledger"],
        "description": "数据新鲜度、任务、告警、服务版本和降级原因",
        "freshness": "服务运行态",
    },
    {
        "id": "northbound",
        "label": "北向持股结构",
        "icon": "pie-chart",
        "primary": False,
        "read_path": "/api/north_holdings?indicator=5日排行&limit=50",
        "description": "持股数量、持股市值、占比与区间变化；不等同停更的日频净买入",
        "freshness": "以实际持股披露日期为准",
    },
    {
        "id": "knowledge",
        "label": "方法论检索",
        "icon": "book-open",
        "primary": False,
        "read_path": "/api/rag_search?q=风险门控",
        "deep_paths": ["/api/industry_chain_history", "/api/research_images"],
        "description": "方法论检索、产业链历史与研报图像证据；不把知识文本冒充行情证据",
        "freshness": "本地知识库索引",
    },
)


def build_catalog() -> dict[str, Any]:
    """Return a JSON-safe catalog with explicit compatibility policy."""
    return {
        "ok": True,
        "contract": API_CATALOG_CONTRACT,
        "service": "quant-web",
        "routing_policy": {
            "canonical": "按业务域使用稳定中性路径；版本号只保留在兼容入口",
            "legacy": "v11/v12 路径继续只读兼容，不再新增页面或业务逻辑",
            "source_of_truth": "统一决策快照和 research_report.v1",
        },
        "domains": [dict(item) for item in API_CATALOG],
    }
