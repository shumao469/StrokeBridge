# Inputs, functions and libraries / 输入、功能与库

## A. 当前可直接使用的功能

| 层 | 输入 | 输出 | 实现库 |
| --- | --- | --- | --- |
| 证据浏览 | 打包汇总JSON | A32/A33分阶段表、A34上限与状态 | Python标准库或离线浏览器JavaScript |
| 一对一身份审计 | 预先定义的合法参考—候选边 | 最大基数匹配/患者联合覆盖 | Python标准库；增广路 |
| 端点错误归类 | 参考/候选on-off、R身份、尺度 | 局部误差或检出/身份失败 | `math`标准库 |
| 有限患者分位数 | 一名患者一分数，保留无穷 | 严格次序统计量 | `math`标准库；不是条件覆盖证明 |
| 时间/缺失合同 | 合成结构JSON | 结构错误/警告；不生成风险 | `datetime`等标准库 |
| 来源核验 | 工作区与manifest | SHA一致/更改/缺失 | `hashlib`、`pathlib` |
| 研究Web演示 | 汇总JSON＋合成示例 | 研究浏览页面，无上传 | Streamlit（可选） |

## B. 未来真实ECG的最小旁车字段（当前不在演示中接收）

每条记录至少应有：`research_patient_id`、`record_id`、`site_id`、原始采集/导出时间与时区、设备型号/软件版本、`sampling_rate_hz`、`n_samples`、导联名称及顺序、物理单位、ADC增益与基线、波形文件SHA、滤波/重建/重采样历史、记录与患者映射证据、标签定义/来源/时间、访问许可和训练曝光说明。

波形格式优先原始可解码DICOM-ECG/SCP/XML或带完整单位的WFDB；也可医院验证过的12导联矩阵及JSON旁车。标注CSV/JSON需同时保存导联、onset/peak/offset的样本坐标及时间坐标、原采样率、符号约定、P/无P/不确定/不可读、标注范围、标注者与复核来源。标注空白不是无P；全局P边界不是V1局部边界；VT/SVT或AF诊断标签不是逐波无P真值。

PTB-XL的官方500Hz发布与100Hz版本要明确区分；文献中采集400Hz、发布上采样500Hz的处理不能被写成设备原生500Hz。不同库不得盲目统一单位或导联顺序；没有官方证据不靠推断放行。

## C. 未来脑心风险合同（只是结构检查）

使用`python -m strokebridge example`获得**虚构**示例。关键字段：

- `patient_id/site`：研究伪名，不上传直接身份；跨资料用医院保管的映射连接。
- `index_time`：预测决策时点；所有特征的`acquired_at`和临床可用`available_at`不得晚于该时点。
- `task`：`future_first_AF`、`future_first_stroke`、`future_recurrent_stroke`三选一；既往史和入组人群分别定义。
- `features.clinical/ecg/brain`：每项保存`name/value/status/acquired_at/available_at`；缺失、不可读、被拒绝分开记。
- `outcome`：目标、裁定状态、目标事件时间、死亡时间、最后确认随访、明确事件阴性随访、来源与裁定者。

日期只有“日”时，不应无依据补零点凑时间；需另行冻结时间分辨率政策。无事件记录不自动为阴性。死亡先于目标事件应按竞争事件明确记录。结构通过不解决患者真实性、入组偏倚、样本量或临床独立验证。

## D. 历史训练/图表库：用途与隔离

| 库 | 历史用途 | 收尾包是否必需 |
| --- | --- | --- |
| PyTorch | ECG分割/有序端点网络、冻结编码器提取 | 否；本包无网络加载或前向 |
| NumPy / SciPy | 信号数组、滤波、窗口和数值运算 | 否；历史方法可能依赖 |
| scikit-learn / joblib | 分类器、预处理、概率校准与保存 | 否；本包无拟合/预测 |
| WFDB / h5py | 公共ECG波形、头信息与HDF5读取 | 否；演示不读原波形 |
| Matplotlib / SVG/PPT工具 | 科学结果和矢量图出版 | 否；图件代码位于上级论文包 |
| Streamlit | 只读研究应用 | 仅运行`streamlit_app.py`需要 |

2026-10-25只读核查的WSL默认解释器为Python3.10.12、NumPy1.26.4、PyTorch2.9.1；不能因此推定所有历史实验均使用该解释器。历史冻结环境与每阶段运行记录优先。本轮不修改既有训练环境。

`requirements.txt`只固定直接UI依赖`streamlit==1.50.0`（PyPI已核验该版本支持Python3.10），不是完整可传递依赖锁，亦未声称本机已安装/验收Streamlit。云部署前在隔离环境解析依赖、记录完整环境和安全扫描，测试通过后再发布。

