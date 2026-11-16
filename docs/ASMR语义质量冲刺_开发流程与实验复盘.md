# ASMR 语义质量冲刺：开发流程与实验复盘（2026-10-30）

> **状态声明**：本文是 2026-10-30 语义冲刺的**历史复盘，不是当前契约**。
> 当前结论以 [项目现状.md](项目现状.md) 的实测矩阵与 [变更历史.md](变更历史.md) §52/§53 为准；旧测试数、旧指纹与旧结论只在本文件/历史章节内有效。

> 性质：项目内复盘文档，不是发布说明。
> 结论先行：E0 + T1 + 环境隔离已交付；T2 提示词未证明更好、未默认；fresh-blind 55% 门槛未达到；本轮成本约 100 元 / 1.2B token，ROI 差。
> 本文记录本轮做过什么、哪些实验有效/无效、哪些值得继续；不把 INVALID 写成 PASS，也不把“文本不同”写成“质量提升”。

---

## 1. 目标与成功标准

原始目标：全面优化 ASMR 翻译的语义质量，按 P0 事实/评测基建 → P1 讨论会 → P2 开发集实验 → P3 实现加固 → P4 盲测 → P5 全量回归推进。

发布门槛（预注册）：

- fresh-blind A/B 胜率 ≥ 55%；
- Wilson 95% 下界 > 50%；
- 结构、QC、延迟不回归；
- 候选 failure 0；
- 单实例、显存 ≤ 7680 MiB。

实际结论：**门槛未达到**；P4 不是 PASS/FAIL，而是 `INVALID_MEASUREMENT_UNREACHABLE`（度量不可达）。

---

## 2. 最终状态

| 项 | 值 |
| --- | --- |
| 主分支 | `main` |
| HEAD | `95135af` `feat(asmr): 合入 ASMR 正式改进与环境隔离` |
| 测试 | **686 tests / 1 skip / OK** |
| 生产模板 | `config/prompts/task_translate_asmr.md`，12,784 B / `CC2B4C82…` |
| 实验模板 | `config/prompts/task_translate_asmr_experimental.md`，5,399 B / `7E589E42…` |
| 默认环境 | production |
| 实验开关 | `--env experimental` + `translate.environment` |
| T2 状态 | 仅实验，未默认；`c1 2a1c65e` 保留为历史 ref |
| P4 | `INVALID_MEASUREMENT_UNREACHABLE` |
| 语义收益 | **未证实** |
| 临时证据 | 冲刺目录 `ASMR_Sprint_20261030` 已清理；本文是保留记录 |

> 已被 §52/§53 推翻：生产模板为 **12,427 B / `A5E05880…`**，全量单测为 **842 例 / OK / 1 skip**（838 基线 + 4 条文档契约护栏）；T1 闭集已从 3 个扩到 5 个。上表数值保留为 2026-10-30 冲刺快照，不再代表当前契约。

提交链：

- E0：`d2e801d`（空行结构修复）
- T1-only：`36471a2` → merge `c31e160`
- 环境隔离：`a0ca92a` → merge `fc2b8c7`
- 项目文档：`db2c098` / `b7a7b13` / `0d01001` / `cd8390f`
- 合入 `main`：`95135af`（父 `a79ae67` + `cd8390f`；旧 main 落后 120 commits，合入为整条工作线，103 files / +24,923−2,035）

未 push、未打 tag、未 bump 版本号。

---

## 3. 开发流程时间线

### P0 事实/评测基建

- 重建项目事实，修正文档与源码不一致。
- 建 fresh-blind 6 文件 / 514 cue，固定 opaque 编号；baseline 与 candidate 隔离。
- 协议 v1.3-4/v1.3-5/v1.3-6：3 角色评委、校准、文本 diff、可行性闸门、冻结清单。
- 关键教训：**先建评测再找候选**，导致评测成本远大于候选实验（见 §8）。

### P1 讨论会/路线图

- 多子 Agent 讨论、方案审计、路线图 v2.1a/b。
- 最终执行候选被收窄为：T2 提示词压缩、T1 三个确定性守卫、E0 结构修复。
- **没有把“管线/算法原型”设为 P3 的硬性交付**；这是本轮最大的执行偏差。

### P2 开发集实验

- G0/G0T：temp0 噪声地板、diff 稳定性。
- R1T：T2 单变量、temp0；文本 diff 111/149 = 74.50%，projected 382 ≥ 360；P1/P2/P3 代理改善；墙钟 0.898×。
- 这些是**开发集代理指标**，不是 blind 语义胜率。

### P3 实现加固

- C1 `2a1c65e`：T2（5,399 B）+ T1；671 tests。
- T1-only `36471a2`：生产模板 12,784 + T1；669 tests；后并入 `c31e160`。
- 环境隔离 `a0ca92a`：通用 `--env` + `EnvironmentProfile`；686 tests；后并入 `fc2b8c7`。

### P4 盲测

- baseline 514：267.851s，121 请求，vram_mib 6003；1 条已知 content failure，按 v1.3-4 保留、不重跑。
- candidate C1 514：0 failure，结构 100%，QC error 0 / actionable 7，245.069s，97 请求，vram_mib 6560；T1 命中 1 条。
- 文本 diff：139 条 identical / 375 条 diff（72.96%）。
- 50-cue 三票 dry-run：仅 8 条 decided；外推 primary 107（95% 57–177）< 360；secondary 107 < 263。
- 结论：`INVALID_MEASUREMENT_UNREACHABLE`；不是候选质量结论，没有胜率。

### forced-choice pilot

- 100 条 diff cue，排除 dryrun50 的 28 条；全新 anchors；no tie。
- 结果：anchors winner 10/10、code 9/10；pairwise agreement 83.3%；`|A−B|` 10.0pp（cue 多数）/9.33pp（300 判定）；decided 100/100；invalid 0。
- 候选侧 53.0%，p=0.368，不显著；63/100 confidence=1；critical 7/100；A 侧胜率 59.1%、B 侧 48.2%，位置效应 +10.9pp。
- 结论：机械 PASS（贴线），但信号在噪声层；**不铺 375**。

### P5 全量回归/交付

- T1-only 1404：
  - base：QC 8/4，700.123s，335 请求，vram_mib 6553；
  - T1 v1：6/5，701.165s，332，6549；
  - T1 v2：**5/5**，705.857s，348，6438。
  - v2 的 5 条 error 全是 base 子集，新增 0；cue61/63 修复；T1 5 accepted / 2 rejected / 0 budget（total 7/14，per_file 2/8）。
  - 残差：5 条 base 已有 duplicate（`Tr2#45`、`曲目6#29`、`自由#13/#445/#506`）+ cue63 检测缺口；绝对零 QC error 不满足。
- 环境隔离 smoke：23 cue，production / experimental / 默认三次均 23 cue、0 failure、QC 0/0；meta 分别为 production/CC2B/12784、experimental/7E58/5399、production/CC2B/12784；默认未加载实验模板。

---

## 4. 实验清单（做过什么、结论是什么）

| ID | 假设/目的 | 方法 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| E0 | 空行会把 1 cue 写成 2 block | 结构修复 + 514/1404 结构检查 | 结构 100%，空行 0 | **保留** |
| T2 | 压缩提示词可提升语义/降 token | 12,784→5,399；dev + 514 blind | dev 代理改善；blind INVALID；pilot 53% 不显著 | **不默认，留实验** |
| T1 | 3 类确定性错误可定点修 | 短重复/称谓/道具臆造 + 预算 | P5 v2 5/5，base 8/4；新增 error 0；61/63 修复 | **保留**（相对无回归） |
| P4 A/B | 514 三评委可判语义优劣 | 3 角色 × 514 | 139 identical/375 diff；decided 107<360 | **INVALID，不是质量结论** |
| forced-choice | 禁 tie 后可判 | 100 diff × 3 | 53% vs 47%，p=0.368；63% conf1 | **噪声，不铺 375** |
| 可行性闸门 | 发评委前机械阻断不可达 | v1.3-6 `feasibility` | 自测 3/3；P4 带 dryrun BLOCK | **值得保留为流程门** |
| 环境隔离 | 生产/实验独立、默认安全 | `--env` + profile + meta | 686 tests；23 cue smoke 三路正确 | **保留** |
| V6 lookahead（本轮前） | 后文参考提升指代 | 真机 A/B | 变慢、回退增多 | **默认关闭**（历史结论） |
| high-confidence 子集 | 只判 conf≥2 的 cue | 未执行 | 外推约 139/375，power 不足 | **未做；不适合发布** |
| 第二候选 | 更激进 T2 | 未执行 | 换候选不解决 tie/位置偏置 | **未做；需新预注册** |
| full 375 | 铺满 diff | 未执行 | 预计 ~1.3M token，仍不可解释 | **未做，不建议** |

---

## 5. 关键数据

### T2 开发集（R1T，temp0）

- 文本 diff：111/149 = 74.50%；projected 382/514 ≥ 360。
- P1 破折号 2.68% → 0；
- P2 无源“也” 3.36% → 2.01%；
- P3 片段补全 10.81% → 2.70%；
- 墙钟 0.898×。

这些是**窄代理**，不是整体语义；开发集 diff 高不代表 blind A/B 可判。

### P4 514

- 139 identical / 375 diff（72.96%）。
- 50-cue 三票：8 decided / 41 tie / 1 no_majority；diff 层 28.6%。
- 外推 primary decided 107（95% 57–177）< 360；V4 nondecided 407 > 154。
- secondary decided_diff 107 < 262.5。
- candidate 工程面：0 failure、结构 100%、QC error 0、actionable 7、245s、97 请求、vram 6560。
- 红旗：b05 cue122 语义不符；7 条“您”语域；T1 1 命中（b03 cue41 <SYNTHETIC_EXAMPLE> acceptable）。

### 校准

- J1/J2/J3：winner 10/10；code 9/10、10/10、10/10；准入 PASS，但带 caveat：首版自述参考过协议 §5.1 的“5 类各 2 条”分布，**不是无瑕疵校准**。
- 正式 P4 未跑；pilot 的 63% confidence=1 说明评委对近义差异不敏感。

---

## 6. 代码/行为变化

### 已合入默认

1. **E0**：在 `postprocess_cue_text`、`validate_cue_translation`、单句裸文本出口折叠连续空行；防止 1 cue → 2 block。
2. **T1**：`<SYNTHETIC_EXAMPLE>` / `<SYNTHETIC_EXAMPLE>` / `<SYNTHETIC_EXAMPLE>`；`audit_retry()` 定点重译；`T1Budget.for_corpus`/`begin_file`：
   - `max_total = max(5, round(total_cues*1%))`（514→5，1404→14）；
   - `per_file = max(2, ceil(file_cues*1%))`（712→8）。
   - `t1_events.jsonl` / `t1_budget.json` 旁路。
3. **环境隔离**：
   - `--env production|experimental`；`translate.environment` 默认 production；
   - `EnvironmentProfile` 按 env+preset 解析模板；
   - experimental 仅 ASMR 覆盖 `task_translate_asmr_experimental`；其它 preset 明确报错、不 fallback；
   - `pipeline_factory/hooks` 是未来整条管线替换的 seam（当前不产生行为）；
   - `translation_meta.json` 记录 environment/task_template/prompt_sha256/prompt_bytes/preset/temperature/model/started_at。

### 未合入默认

- T2 提示词：只在 experimental 模板中；`c1` 分支保留历史。
- 更激进的管线/两遍式/rerank：**没有实现**。

---

## 7. 为什么 T2 没判别出来

1. **只换 prompt，没换管线/算法**：同一模型、权重、窗口、解码；prompt 只能改风格/规则，不能新增能力。
2. **文本差异大，语义差异小**：73% cue 文本不同，但多数是标点、语气词、同义词、词序；评委给 confidence=1 是诚实。
3. **baseline 已经很强**：514 QC error 0，只有 1 条已知 failure；随机样本里“可修的错误”太少，效果被稀释。
4. **评委 tie 政策 + 三票多数**：3 个纯文本 AI 评委，对近义差异大量 tie；必须 2 人同向，decided 上不去。
5. **forced-choice 把“无差别”变“抛硬币”**：53% p=0.368；63% conf1；位置偏置 +10.9pp。
6. **ASMR 关键维度在音频**：SRT-only 评委看不到气息/节奏/贴耳感；文本近义时更判不出。
7. **定向错误太少**：T2 的真实红旗（b05 语义不符、7 条“您”）是稀疏的，不足以支撑 55%；也可能局部更差。

---

## 8. 成本与 ROI（诚实账）

- 本轮约 **100 元 / 1.2B token**。
- 大头不是 GPU 翻译（candidate 514 约 4 分钟、P5 1404 约 12 分钟），而是：
  - P4 评委/校准/dry-run/pilot；
  - 多 Agent 互审、长文档/源码上下文；
  - 反复对齐口径与文档。
- 产出：
  - E0（真实结构修复）；
  - T1（少量确定性修复，P5 5/5 vs 8/4，新增 0）；
  - 环境隔离（工程能力）；
  - 阴性证据与可行性闸门。
- **结论：ROI 差。** E0/T1 本可以用远低预算完成；大部分预算花在“证明 prompt 微调不可判”和流程/文档上。
- 主要责任在调度：没有硬预算、没有 50-cue 早停、没有把管线原型设为硬门、没有挑战“prompt 能拿 55%”的前提。

---

## 9. 哪些值得做（按优先级）

### P1｜两遍式翻译（draft → critique/repair）

- 同一模型，第一遍初译，第二遍只对高风险 cue/片段做批评+修复；
- 优点：不改模型/依赖，8GB 可跑；能针对语义而非风格；
- 成本：约 2× 推理；
- 验证：50–100 条**错误富集** cue，先看 critical 修复率与盲 A/B，不直接上 514；
- 这是本轮最该做但没做的。

### P2｜候选生成 + 重排

- 同一 cue 生成 2–3 个候选，用规则/同一模型/小模型选一个；
- 适合处理否定、主客、称谓、道具等可判别错误；
- 需要定义选择规则与成本上限；
- 先 50 条 pilot。

### P3｜上下文/窗口/实体管线重构

- overlap、跨窗实体、源文预处理/断句、长对话的说话人/指代；
- V5/V6/P2a 已有部分能力，但没有系统重构成新管线；
- 这是“管线强化”的正路，不是 prompt 微调。

### P4｜定向评测，而不是随机 514

- 用已知错误类型/taxonomy 富集 cue；
- 加 reference 或 human/audio 评委；
- 先 50–100 条 cheap pilot，有信号再扩；
- 随机 514 + 纯文本 AI 评委对“近义/风格”差异不敏感，本轮已证明。

### P5｜模型/量化权衡

- Q6/Q5/Q4、profile/ctx 的 trade-off；
- 更大的模型未必进 8GB；
- 先小样本质量/延迟对比，不做全量。

### P6｜T1 小修补

- cue63 检测缺口（源 >8、译文 <12 的 duplicate）；
- 5 条 base 已有 duplicate；
- 环境名三处来源统一；
- `--env` 只覆盖 translate；batch/其它命令未路由；如要通用，提升顶层 `runtime.environment` 或 app 注入；
- 这些是工程小项，不是语义大招。

---

## 10. 哪些不值得再做

- **把 prompt 大改当成主方案**：本轮已证明 73% 文本 diff 仍不可判；横向改写不等于纵向提升。
- **forced-choice 硬压近义句**：53% p=0.368、63% conf1、位置偏置；只会把 tie 换成噪声。
- **full 514/3 评委在 pilot 前**：1.3M token 买不可解释结果。
- **high-confidence 子集当发布**：约 139/375，power 不足，只能描述。
- **第二候选继续同一 prompt 思路**：不解决 tie/位置偏置/语义差异小。
- **多 Agent 互审 + 多份报告**：本轮主要成本之一；一份 source of truth 足够。

---

## 11. 未验证/边界

- T2 的正式 375 diff forced-choice 未跑；pilot 显示噪声，故停止。
- P4 514 的正式评委未跑；按可行性闸门 BLOCK，未消耗 1542 判定。
- ASMR 音频听感/人工盲评未做；SRT-only 评委看不到声音。
- 其它 preset 的 experimental 未实现（明确报错，不 fallback）。
- `translate.environment` 只覆盖 translate；batch/其它命令未路由。
- T1 绝对零 QC error 未达：5 条 base duplicate + cue63 缺口。
- 环境名三处来源未统一；`translation_meta.json` 对所有 `--out-dir` 写，`"w"` 覆盖。
- 未 push、未 tag、未 bump 版本号。

---

## 12. 复现/使用

默认生产：

```powershell
python -m vanitas translate "xxx.srt" --preset asmr --out-dir runs\ab\prod
```

实验：

```powershell
python -m vanitas translate "xxx.srt" --preset asmr --env experimental --out-dir runs\ab\exp
```

配置：

```json
"translate": { "environment": "production" }
```

`translation_meta.json` 用于识别 environment/task_template/prompt SHA/bytes。

测试：

```powershell
python -m unittest discover -s tests
```

当前应 **686 tests / 1 skip / OK**。

---

## 13. 一句话结论

**默认生产：结构更好、少量确定性错误更少，整体语义与之前基本持平；T2 未证明更好，55% 未达到；本轮工程交付可用，但投入产出不理想。下一步应做两遍式/候选重排/上下文重构 + 定向评测，并用硬预算和 50-cue 早停控制成本。**
