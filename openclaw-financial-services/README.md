# 🏦 Financial Services Skills for OpenClaw

> Turn [OpenClaw](https://openclaw.ai) into a financial services specialist — investment banking, equity research, private equity, and wealth management.

Converted from [Anthropic's Claude Financial Services Plugins](https://github.com/anthropics/financial-services-plugins) to the OpenClaw skill format.

## ✨ What's Included

**100 skills** covering end-to-end financial workflows:

| Module | Auto Skills | Commands | Description |
|--------|:-----------:|:--------:|-------------|
| **Financial Analysis** | 9 | 8 | DCF, comps, LBO, 3-statement models, deck QC |
| **Equity Research** | 9 | 9 | Earnings analysis, initiating coverage, morning notes |
| **Investment Banking** | 9 | 7 | CIM drafting, buyer lists, merger models, pitch decks |
| **Private Equity** | 9 | 9 | Deal sourcing, due diligence, IC memos, portfolio monitoring |
| **Wealth Management** | 6 | 6 | Client reviews, financial plans, rebalancing, tax-loss harvesting |
| **LSEG** | 8 | 8 | Bonds, FX carry, options vol, macro rates (via LSEG data) |
| **S&P Global** | 3 | — | Tear sheets, earnings previews, funding digests |
| **Total** | **53** | **47** | **100 skills** |

Plus **12 MCP data source connectors** for institutional-grade financial data.

## 🚀 Installation

```bash
# 1. Clone this repo
git clone https://github.com/d-wwei/openclaw-financial-services.git

# 2. Copy skills to OpenClaw
cp -r openclaw-financial-services/skills/ ~/.openclaw/skills/

# 3. Merge MCP config into your openclaw.json
# Open openclaw.json and merge the "skills" and "mcp" sections
# into your existing ~/.openclaw/openclaw.json

# 4. Restart OpenClaw Gateway
```

## 💡 Usage

### Auto-triggered Skills (🧠)

Just talk naturally — skills activate when the topic matches:

```
"帮我做一个苹果公司的 DCF 估值模型"     → fsi-fa-dcf-model
"分析特斯拉 Q3 财报"                    → fsi-er-earnings-analysis
"帮我起草一份 CIM"                      → fsi-ib-cim-builder
"筛选一下这份 CIM 是否符合我们的投资策略"  → fsi-pe-deal-screening
"准备一下张总的季度回顾会议"              → fsi-wm-client-review-prep
```

### User-invocable Commands (🔧)

Use as slash commands in supported channels:

```
/fsi-fa-cmd-comps NVIDIA          # 可比公司分析
/fsi-fa-cmd-dcf Tesla             # DCF 估值
/fsi-er-cmd-earnings AAPL Q4      # 季度财报分析
/fsi-pe-cmd-ic-memo ProjectAlpha  # IC 备忘录
/fsi-wm-cmd-rebalance ClientName  # 投资组合再平衡
```

## 🔌 MCP Data Sources

These MCP servers provide institutional-grade financial data. API keys or subscriptions may be required.

| Provider | URL | Data |
|----------|-----|------|
| Daloopa | `https://mcp.daloopa.com/server/mcp` | Standardized financials |
| Morningstar | `https://mcp.morningstar.com/mcp` | Fund/stock ratings |
| S&P Global | `https://kfinance.kensho.com/integrations/mcp` | Capital IQ data |
| FactSet | `https://mcp.factset.com/mcp` | Financial data terminal |
| Moody's | `https://api.moodys.com/genai-ready-data/m1/mcp` | Credit ratings |
| MT Newswires | `https://vast-mcp.blueskyapi.com/mtnewswires` | Real-time news |
| Aiera | `https://mcp-pub.aiera.com` | Earnings transcripts |
| LSEG | `https://api.analytics.lseg.com/lfa/mcp` | Bonds/FX/Macro |
| PitchBook | `https://premium.mcp.pitchbook.com/mcp` | PE/VC deal data |
| Chronograph | `https://ai.chronograph.pe/mcp` | PE portfolio data |
| Egnyte | `https://mcp-server.egnyte.com/mcp` | Document management |

To configure authentication, add headers to your `openclaw.json`:

```json
{
  "mcp": {
    "servers": {
      "factset": {
        "type": "http",
        "url": "https://mcp.factset.com/mcp",
        "headers": {
          "Authorization": "Bearer YOUR_API_KEY"
        }
      }
    }
  }
}
```

## 📄 License

[Apache License 2.0](./LICENSE)

## ⚠️ Disclaimer

These skills assist with financial workflows but do not provide financial or investing advice. Always verify conclusions with qualified financial professionals. AI-generated analysis should be reviewed before being relied upon for financial or investment decisions.

## 🙏 Credits

- Original plugins by [Anthropic](https://github.com/anthropics/financial-services-plugins)
- Partner plugins by [LSEG](https://www.lseg.com/) and [S&P Global](https://www.spglobal.com/)
- Converted for [OpenClaw](https://openclaw.ai)
