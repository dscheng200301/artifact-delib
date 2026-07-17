# HistoDelib 论文写作细节规范

Disagreement-Triggered Bimodal Deliberation with Deferred Cross-Modal Adjudication for Historical Image-Caption Verification

## 目录

1. 文档目的与边界
2. 任务定义
3. 研究问题
4. 方法章节蓝图
5. API-only 系统记录
6. 基线章节规范
7. 消融实验规范
8. 评价指标与公式
9. 结果表与图规范
10. 失败案例分析
11. 局限性与伦理
12. 正式实验前置条件
13. 复现与数字一致性
14. 论文完成前清单

## 1. 文档目的与边界

- 本文件是 HistoDelib 后续论文写作规范，不是正式论文。它只约束未来的写作、结果导出和复现记录。
- 正式数据、正式实验、统计检验和结果数字完成前，所有未完成内容写为 NOT_RUN 或待实验完成后填写；无正式数据支持的指标写为 N/A。
- SYNTHETIC_FIXTURE 与 NOT_FOR_RESEARCH_RESULTS 只能描述流程验证，不能进入正式表格、图或结论。

## 2. 任务定义

- 输入为历史图像 I 与文字说明 T，输出为 TRUE、MISCAPTIONED 或 OUT_OF_CONTEXT。
- TRUE 表示主要历史对象、人物、时间、地点、事件和归属相互一致。
- MISCAPTIONED 表示围绕同一对象或主题但至少一个关键历史属性错误，属于细粒度错误。
- OUT_OF_CONTEXT 表示图像与文字来自不同对象、人物、事件或整体历史语境，属于整体错配。
- 内部 INSUFFICIENT_EVIDENCE 是流程状态，不自动等同于正式标签。

## 3. 研究问题

- RQ1 模态隔离：Text Agent 与 Image Agent 的独立证据是否减少跨模态泄漏，并保持可审计性。
- RQ2 分歧触发交互：Light Relation Probe、定向重审和受控质询是否只在风险出现时增加交互。
- RQ3 动态执行效率：Router 与 Deferred Judge 是否在准确性、细粒度错误识别和 Token 代价之间形成可分析权衡。
- 每个研究问题都必须对应必要基线、单因素消融、主要指标、失败案例和不能预设的结论。

## 4. 方法章节蓝图

- Text Agent 只接收文字说明，提取声明、时间、地点、身份、事件和归属属性；不验证文字本身的真实度。
- Image Agent 只接收图像，记录整体场景、铭文、局部细节和视觉证据不足；不读取 Caption。
- Light Relation Probe 只输出短风险特征，为 Router 提供 modality_disagreement、temporal_conflict、location_conflict、identity_conflict 和 unreadable_glyph 等信号。
- Rule Router 低成本、可解释；API Router 只读取结构化 Agent 证据，输出经过 Schema 验证的路由决策。
- 定向重审使用 Text、Glyph、Panor 和 Patch 视图；每个视图必须记录触发原因和产生的增量证据。
- Controlled Cross-Examination 使用固定最大轮数、ABSTAIN、稳定状态、无新增证据和预算停止条件。
- Deferred Judge 先保存盲初判，再读取结构化证据，输出 KEEP、REVISE 或 ABSTAIN；不等同于隐藏思维链。

## 5. API-only 系统记录

- 每次正式调用记录 Provider、Model Name、Model Version or Snapshot、API Date、Temperature、Max Output Tokens、Image Detail Setting、Retry Policy 和 Prompt Version。
- 记录 Input Tokens、Output Tokens、Total Tokens、Latency、Cache State、Request ID、错误类型和估算成本；Authorization 与 API key 永不进入日志。
- API 版本漂移、非确定性、图像 Token 不透明、服务商价格变化和网络延迟必须写入局限性。

## 6. 基线章节规范

- 至少覆盖 Text-only LLM、Image-only VLM、Direct VLM、Structured Reasoning VLM、Self-Consistency、Self-Reflection、Sequential Context-Veracity、Fixed Multi-Perspective、Generic MAD、Always-Full、HistoDelib-Rule 和 HistoDelib-API-Router。
- 适配其他论文思想时使用 adapted baseline；未完整使用原论文代码与数据时不得称为严格复现。
- Self-Consistency 的全部采样 Token、Generic MAD 的全部轮次 Token 和 Always-Full 的完整调用都必须计入总量。

## 7. 消融实验规范

- 主消融包括 Full、w/o Modality Isolation、w/o Targeted Reinspection、w/o Cross-Examination、Free Debate、w/o Deferred Judge 和 w/o Router / Always-Full。
- 附加策略包括 N=0、Fixed N=1、Adaptive N<=2、Fixed N=2、Fixed N=3、Rule Router 和 API Router。
- 一次消融尽量只改变一个因素；Prompt、模型、数据、随机种子和其它流程变量保持不变。
- 消融结果只描述观察到的差异，不预设某个组件一定提升性能。

## 8. 评价指标与公式

- 核心指标为 Accuracy、Macro-F1、MISCAPTIONED-F1、Average Tokens 和 Token Saving；Macro-F1 是主指标。
- Token Saving = (AlwaysFullTokens - MethodTokens) / AlwaysFullTokens * 100%。负值表示 Token 增加。
- Correction Rate = 初判错误且终判正确的样本数 / 初判错误样本数。
- Harm Rate = 初判正确且终判错误的样本数 / 初判正确样本数。
- Token 减少不等同于延迟或美元成本同比减少；不同 API 的 Token 计算方式可能不可完全比较。

## 9. 结果表与图规范

- Table 1 至 Table 3 只定义列与数据来源；缺失值使用 --，不得手工复制数字。
- 图形可包括系统架构、三分类任务示例、Macro-F1 与 Average Tokens Pareto、消融相对 Full 的变化、质询轮数与路由分布。
- 所有数字只能从结构化预测文件自动导出；摘要、正文、表格、图和结论必须通过一致性检查。
- 未有统计检验不得写显著；未检查所有公平基线不得写最优；没有严格同设置证据不得写 SOTA。

## 10. 失败案例分析

- 至少覆盖视觉证据不足、文字说明过度具体、图片文字不可读、历史年代线索模糊、人物身份相似、地点建筑相似、共同错误、高风险质询修正、Router 错路和 API 解析失败。
- 每个案例记录输入、真实标签、初判、路由、重审、质询、终判、Token、错误原因和改进方向。
- 不得只挑选成功案例，也不得把内部 Agent 证据当作外部历史资料。

## 11. 局限性与伦理

- 内部图文证据无法替代外部历史资料；图像可能缺少验证线索；机构元数据也可能错误。
- API 模型版本、视觉年代判断、文化偏差、共同错误传播、动态尾延迟和 Token 统计差异都影响复现。
- 正式论文必须披露图像来源、授权、人物与敏感场景、人工标注、API 数据接收、缓存删除和生成式 AI 参与范围。

## 12. 正式实验前置条件

- 正式数据集确认、授权确认、Schema 验证、原图分组、Train/Validation/Test 冻结、Prompt 版本冻结、基础模型与 API 版本记录。
- Always-Full、全部基线、核心消融、Token 统计、缓存、预算、随机种子、重复次数、错误恢复和结果复核全部完成后，才可填写结果。
- 测试集不得用于 Prompt 选择或阈值调节。

## 13. 复现与数字一致性

- 记录 Python 3.12、Conda 环境、Git commit、模型供应商、模型版本、API 日期、Prompt hash、温度、输出 Token 上限、图像处理、缓存、预算、数据版本、随机种子和重复次数。
- 数字只能从结构化结果文件自动生成 LaTeX 宏、表格和图；检查百分比与百分点、Token Saving 基准、随机种子平均和统计声明。
- 生成 PDF 前检查目录、页码、中文字体、表格、代码块、缺失章节、结果数字扫描和匿名信息。

## 14. 论文完成前清单

- 正式数据授权已确认；所有主基线和核心消融已运行；Token 已复核；三个随机种子或预先声明的重复方案已完成；置信区间和失败案例已记录。
- Fixture 结果未进入正文；数字来自脚本；结论没有夸大；局限性、伦理和会议要求已复核。
- 当前工程阶段的最终状态仍应显示 NOT_RUN 或 N/A，直到上述条件被真实证据满足。
