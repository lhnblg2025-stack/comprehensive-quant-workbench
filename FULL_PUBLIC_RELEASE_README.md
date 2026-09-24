# A 股量化工作台：全量代码脱密公开版

发布日期：2026-09-24

本分支公开工作台的代码资产，包括：

- `quant_system/`：量化研究、回测、因子、风控、执行适配和 RAG 实现
- `quant_web/`、`quant_platform/`：Web 与平台层
- `scripts/`、`tests/`、部署文件、配置模板和项目文档
- `openclaw-financial-services/skills/`：OpenClaw skills
- `skills/knowledge_skills/`、`skills/recovery_skills/`：知识技能文本
- `rag/`：RAG 实现与索引元数据

为脱密和版权边界，未纳入：密钥/证书、`.env`、数据库、行情与财务数据、交易/账户记录、运行日志、缓存、模型二进制、Git 历史、恢复目录、原始书籍文件和个人文档。RAG 向量索引需使用公开数据按实现代码重新构建。

运行时请复制 `deploy/quant.env.example` 等模板，并通过环境变量注入密钥。
