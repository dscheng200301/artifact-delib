# Generate HistoDelib Paper Writing Specification PDF

Generate only a future-facing Chinese writing-specification PDF for HistoDelib. This task is not the formal paper and must not run models, search/download datasets, or fill any numerical result.

## Non-negotiable rules

- 不得编造任何实验结果、准确率、Macro-F1、Token Saving、显著性、SOTA 或投稿结论。
- 所有未完成实验标记为 `NOT_RUN`；没有正式数据支持的指标标记为 `N/A`；待填内容写作“待实验完成后填写”。
- Fixture 输出永远标记为 `SYNTHETIC_FIXTURE` 和 `NOT_FOR_RESEARCH_RESULTS`，不得进入正式表格或结论。
- 使用本地免费工具生成可复现 PDF，并在生成后检查页数、中文字体、目录、页码、表格和代码块是否完整；不得使用付费文档服务。

## Required sections

1. **任务定义**：输入为历史图片与文字说明；输出 `TRUE`、`MISCAPTIONED`、`OUT_OF_CONTEXT`。说明 MISCAPTIONED 是细粒度属性错误、OOC 是整体语境错配。
2. **研究问题**：模态隔离、分歧触发的定向重审与受控质询、动态执行与 Token 效率。每个问题列出所需基线、消融与指标，不预设结论。
3. **方法章节**：逐项说明 Text Agent、Image Agent、Light Relation Probe、Rule/API Router、Text/Glyph/Panor/Patch Recheck、受控交叉质询、停止准则、Deferred Judge、缓存、预算和 Token 统计的目的、输入、输出、失败模式和对应消融。
4. **API-only 系统**：记录 Provider、Model Name/Version、API Date、Temperature、Max Output Tokens、Image Detail、Retry Policy 与 Prompt Version；说明 API 可变性和 Token 可比性限制。
5. **基线与消融实验**：覆盖 Text-only、Image-only、Direct VLM、Structured Reasoning、Self-Consistency、Self-Reflection、Sequential Context、Fixed Multi-Perspective、Generic MAD、Always-Full、HistoDelib Rule/API Router；说明所有调用 Token 均计入总量。
6. **评价指标**：Accuracy、Macro-F1、MISCAPTIONED-F1、Average Tokens、Token Saving、Correction Rate、Harm Rate，写明 Token Saving 以 Always-Full 为基准且 Token 减少不等同延迟或美元成本同比减少。
7. **建议表图与结果写作模板**：只定义列、数据来源和 `--` 缺失值规则；不给数字、不写结论。说明表图只可由结构化预测文件导出。
8. **失败案例、局限性、伦理、复现与会议版本侧重点**：明确 API、历史证据、授权、偏差和版本漂移的限制。
9. **正式实验前与论文完成前检查清单**：数据授权、冻结、版本、缓存、预算、种子、复现、数字一致性和匿名检查。

Deliver Markdown source, PDF path, page count, render/inspection result, font warnings, missing-section check, numerical-result scan, and Git commit. Do not execute this PDF build until the engineering project is stable and the user explicitly requests it.
