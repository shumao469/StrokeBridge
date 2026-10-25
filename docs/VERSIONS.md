# Version map / 版本与用途

这里的命名是版本索引别名，不追溯改写旧协议或失败状态。本地冻结原文件未移动或覆盖。公开副本已去除本机绝对路径/账号标签；精确原来源散列见`provenance/source_inventory.json`，导出散列见`ARCHIVE_MANIFEST.json`。

`archive/phase_*/code/`保存99份历史源码，其中22份去除了本机绝对路径/账号标签，不能声称这些改过的公开副本与冻结原件字节一致；原/导出SHA及转换状态见`ARCHIVE_MANIFEST.json`。**这些副本只是档案，不是独立可运行发布模型**：缺少旧工作区的数据/权重/跨阶段依赖，且旧作业窗口、预算及停止状态不被激活。严禁直接执行归档中的训练、恢复或下载入口。主入口仍是`python -m strokebridge`。

| 原阶段 | 收尾别名 | 功能 | 保留与限制 |
| --- | --- | --- | --- |
| A17R | `AF-ID/A17R/patient_split_baseline` | AF标签、患者划分、统一预处理与多组特征 | 当下AF识别；历史native500名称不是采集500Hz事实 |
| A24 | `AF-ID/A24/encoder_controls_refits` | 第二编码器、检出器/错位负控、患者级重拟合 | 原R/N/E恢复、停止和预算均保留；不自动续跑 |
| A25 | `P-MeasureAudit/A25/partial_label_and_missingness` | 部分标注、可靠拒绝、缺失干预与源核验 | 独立参考和形态增量未补齐 |
| A26 | `P-MeasureAudit/A26/patient_calibration` | 患者边界校准和部分标签监督 | 双种子及实用门未共同通过 |
| A27 | `P-MeasureAudit/A27/coverage_attribution` | 连续/R锚定拆解与匹配覆盖比较 | 不以全拒绝基线宣称实际收益 |
| A28 | `P-MeasureAudit/A28/zero_fit_error_decomposition` | 漏检、身份、边界、区间错误拆分 | 仅审计，不构成新模型 |
| A29 | `P-MeasureAudit/A29/identity_unique_box` | 禁止跨心搏借框及候选复用 | 身份合同修正不等于覆盖达标 |
| A30 | `P-MeasureAudit/A30/structured_targets_tail` | 正/空结构化目标与尾部端点监督 | 实验版本，未放行临床测量 |
| A31 | `P-MeasureAudit/A31/report_alignment_audit` | 原始/最终候选及窗截断核查 | 机制诊断，非新训练 |
| A32 | `P-MeasureAudit/A32/patient_multibeat_experimental` | 整患者多心搏监督；R1批次记账修复 | 20fits160epoch完成；测量门失败；data/95未删 |
| A33 | `P-MeasureAudit/A33/zero_fit_factorial_audit` | 2×2有序窗/连续候选因子、oracle容量 | 0fits；16个必要条件未通过；无策略放行 |
| A34 | `P-MeasureAudit/A34/endpoint_only_nonadmission` | 端点模块训练前支持/阈值/校准上限 | **0fits：科学不准入，不是训练成功模型** |
| 本目录 | `StrokeBridge/v0.1/research_closeout` | 无拟合可移植审计合同、汇总和演示 | 12单测＋原30合成合同通过；无临床预测 |

## 从档案复核，不直接重启

1. 先运行`verify-sources`核对原代码未变。
2. 阅读目标阶段`protocol.txt`、`budget.json`、`completion.json`、`post_review/`及`STOP`/`stop.json`，区分最终回执和中间失败。
3. 根据`result_sources.json`对照收尾汇总；需要计算不确定性时保留患者为重采样单位。
4. A32的W1与A33的F00不是同一个政策，不能直接混称。A33统一源池、身份域、阈值及校准规则后才做因子比较。
5. A34的假想“禁用修正C0”指同父模型F01中的端点修正关闭臂，不是自动继承A32旧C0。A34从未训练，不能补写模型性能。
6. 历史训练代码存在跨阶段依赖；这些脚本不作为本目录入口，不可在旧截止/预算下直接执行。任何新拟合须另立有界方案并获准。

