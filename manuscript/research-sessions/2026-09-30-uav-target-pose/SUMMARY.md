# 研究会话：无人机跨尺度靶标位姿估计

日期：2026-09-30。任务：定向综述，服务中文期刊论文引言与相关工作；不宣称系统综述穷尽性。
使用 research-superpowers 检索、筛选、全文提取、引文追踪流程。用户已确认执行方案。
研究状态：本轮定向综述与前两节初稿已完成；筛选分数表示相关性，不代表论文质量。

## 查询记录（批次 M1）
- Fast Explicit Diffusion accelerated features AKAZE（CVF定向检索后补原文作者站点）
- contour texture pose tracking planar object hybrid features
- 无人机 多层标记 视觉定位 自主降落 位姿 航空学报
- recursive ArUco multi scale UAV landing 2026（arXiv定向）

主要来源：期刊/会议与作者原文、arXiv；聚合搜索只用于发现。中文综述仅作线索，不把二手摘要当原始机制证据。
逐篇结果已综合写入 papers-reviewed.json。联系邮箱仅用于开放获取服务请求，不写入研究产物。

## 执行结果与统计口径

| 项目 | 结果 | 口径 |
|---|---:|---|
| 去重候选 | 67 | 三个主题批次63项加主代理4项新记录；正式稿/预印本按同一工作合并 |
| 相关性评分≥7 | 53 | 任务0–3、机制0–4、比较价值0–3；没有人工标注测试集，不报告筛选准确率 |
| 原文相关章节核对 | 36 | 已核对方法、限制或实验设置的相关段落；不是逐字穷读全文或审计全部实验 |
| 原文可得 | 38 | 包含2项只作为背景、未作深读的全文；与上述36项分别统计 |
| 两章引用 | 25 | 22项有相关原文或基础全文来源，3项仅用原始摘要明述的方法类别 |
| 核实引文边 | 34 | 来源参考文献/正文真实关联，保留方向和节页；不根据主题相似性造边 |
| 保存查询记录 | 134 | 批次中的真实查询条目，可能重复；不是数据库覆盖率或134个独立检索主题 |
| 本地PDF归档 | 10 | 合法公开核心原文，核对PDF签名和SHA-256；清单位于papers/archive-manifest.json |
| 正文DOI注册元数据成功查询 | 21 | 24项DOI中21项Crossref访问成功；其余3项已由原始出版/会议记录补齐，另有1项arXiv报告 |

主要通过网络搜索发现原始研究，再核对出版社、会议、作者及机构原文。此次没有直接执行 Scopus、Web of Science 或 IEEE Xplore 数据库API检索，不能据此声称覆盖这些数据库的全部记录。检索覆盖经典工作及已发现的2026工作，截止2026-09-30；不是按预注册协议实施的系统综述。

## 文献筛选与引文追踪

独立批次为跨尺度/分层标记、轮廓纹理/恢复、平面候选/尺度可靠性。分数达到7的条目优先读原文；基础技术可作为例外保留。评分低、偏离任务或只有元数据的条目仍保留在全部记录，不将其当作机制缺失的证据。

真实引文追踪包括 Lee→Flexible Layouts/UcoSLAM，Recursive ArUco→Fractal/Embedded ArUco/层级标记，AprilTag2→AprilTag，STag→RUNE/ArUco，Ch’ng→IPPE/Schweighofer/Wu，以及中文复合地标论文→EPnP/Lynen。记录最大追踪深度不超过2；边表示源论文引用目标论文，与本文组织观点的逻辑链不同。

## 关键结果：贡献必须收敛

1. 分级视觉定位、尺度切换和失败判断已有直接无人机任务先例，尤其Yuan2018、Lv2024和Lee2024。
2. Fractal2019原文既联合内外层角点，也在无完整标记时利用历史FAST恢复。不能照搬后续论文关于“只依赖中心”的简化描述。
3. 边缘与纹理联合、自然与人工特征联合以及条件性回退均已有先例。本文更合理的关注点是特定物理模板上的尺度/可见性调度与同帧求解恢复组织。
4. Wu2012/2014已有时间预测选解及面积相关误差模型。本文跨度门限是验收规则，不等价于协方差模型；IPPE候选与低重投影误差也不保证真实位姿唯一正确。
5. 历史缓冲、延迟更新和重传播是既有融合技术，ESKF适配不列作新的滤波理论。

据此，前两节按“两项主要工作＋配套候选验收”撰写。任务差异和流程组合仍需实验支持，不以未找到完全相同实现证明原创性。

## 证据限制与版本

- 正文摘要级条目：邢伯阳等2019、ArUco2014、Pressigout2007。只使用其出版摘要明确的方法类别，不引用性能数字，不评价未读细节。
- Schweighofer2006、FSCGE2024、PACMAN2026等受限全文候选保留待补状态。已用用户提供的真实邮箱查询开放获取；未绕过访问限制，未把邮箱写入产物。
- Lee使用2024正式DOI，阅读作者arXiv稿；Recursive ArUco使用2026正式期刊DOI，相关章节来源为作者预印本。HybridPose不搬用会议稿与更新稿冲突的实验数字。
- HybridPose书目采用CVF排版431–440，Crossref/IEEE注册页428–437保留在原始元数据中；CALTag按Eurographics正式记录补齐会议类别与页码。Yuan2018注册page字段为截断文章号，省略该字段，不伪造页码。
- 本轮未重新审计源码、复算真实实验或验证飞行闭环。第三节中的“五个连续距离状态”“超分辨率恢复”“全局最优”“所有候选统一校验”“消除冻结”等表述需后续按实现与实验修正。

## 交付文件

- [引言](C:/NEW_Steretracker-master/manuscript/第1章_引言.md)
- [相关工作](C:/NEW_Steretracker-master/manuscript/第2章_相关工作.md)
- [共享参考文献](C:/NEW_Steretracker-master/manuscript/参考文献_引言与相关工作.md)与[references.bib](C:/NEW_Steretracker-master/manuscript/references.bib)
- [文献证据与贡献边界](C:/NEW_Steretracker-master/manuscript/文献证据与贡献边界.md)：近邻工作矩阵、贡献—实验映射及第三节衔接建议
- [全部候选](C:/NEW_Steretracker-master/manuscript/research-sessions/2026-09-30-uav-target-pose/papers-reviewed.json)、[相关候选](C:/NEW_Steretracker-master/manuscript/research-sessions/2026-09-30-uav-target-pose/relevant-papers.json)、[引文网络](C:/NEW_Steretracker-master/manuscript/research-sessions/2026-09-30-uav-target-pose/citations/citation-graph.json)

使用的主要技能：[Getting Started](C:/NEW_Steretracker-master/.codex/skills/research-superpower-main/skills/getting-started/SKILL.md)、[Searching Scientific Literature](C:/NEW_Steretracker-master/.codex/skills/research-superpower-main/skills/research/searching-literature/SKILL.md)、[Evaluating Paper Relevance](C:/NEW_Steretracker-master/.codex/skills/research-superpower-main/skills/research/evaluating-paper-relevance/SKILL.md)、[Finding Open Access Papers](C:/NEW_Steretracker-master/.codex/skills/research-superpower-main/skills/research/finding-open-access-papers/SKILL.md)、[Traversing Citation Networks](C:/NEW_Steretracker-master/.codex/skills/research-superpower-main/skills/research/traversing-citations/SKILL.md)、[Subagent-Driven Literature Review](C:/NEW_Steretracker-master/.codex/skills/research-superpower-main/skills/research/subagent-driven-review/SKILL.md)。贡献审查沿用[scientific-critical-thinking](C:/NEW_Steretracker-master/.codex/skills/claude-scientific-writer-main/skills/scientific-critical-thinking/SKILL.md)。
