# HotpotQA ReAct Agent 项目

复现并改进 ReAct（HotpotQA fullwiki 设定）：Direct / CoT / ReAct 三 baseline → 失败归因 → 逐项消融 → held-out 验证。

## 协议红线（改了就会破坏与论文及组间的可比性）

- 抽样：`random.Random(233).shuffle(range(7405))`，调参集 `[0:500]`，held-out `[500:1000]`
- 评测：只使用 `src/hotpot_evaluate_v1.py` 的官方 normalize / EM / F1，不得自行重写
- ReAct loop：最多 7 步、bad-call 重试一次、循环结束强制 `finish[]`——对齐官方 notebook
- 模型：`deepseek-v4-flash`，非思考模式（thinking 模式下 temperature 无效），temperature=0

## 结构

- `src/hotpot_evaluate_v1.py` 官方评测脚本（只读引用）
- `src/llm.py` DeepSeek client（key 放 `.env` 的 `DEEPSEEK_API_KEY`）
- `src/wiki_env.py` 离线 WikiEnv（search/lookup/finish，观测格式对齐官方 wikienv.py）
- `src/prompts.py` 官方 few-shot prompt（数据在 `prompts/prompts_naive.json`）
- `src/agents.py` 类层级：BaseAgent → Direct/CoT/ReAct → ActOnly/Memory/RAG/Fallback(v1,v2)/Concise/SelfCheck(+strict)/CotSc/ContentFallback/ReActSC；ReActAgent.run 含 memory 钩子、verify_finish 门与 on_forced_finish 钩子（默认均不改变行为，保持 baseline 逐字节一致）；react/actonly 记录 steps_detail
- `src/run_eval.py` 评测 CLI（--env base|rag|fallback|fallback2|content，--prompt base|concise，--selfcheck[--strict]，--cotsc，--sc，--resume），结果落 `results/*.jsonl` + `*_summary.json`
- `scripts/build_wiki_index.py` 从 wiki dump 构建 sqlite 索引（写 .tmp 后原子 rename；`--resume` 断点续建）
- `scripts/build_content_index.py` 建内容级索引 `data/wiki_content.db`（每页首段 FTS5+bm25，content 臂用，2.7GB）
- `scripts/replay_trace.py` trace 回放与失败信号统计（--list/--fail/--signals/--id/--diff）
- `scripts/build_memory.py` 从调参集 trace 挖掘记忆库（cases 规则挖掘 + lessons LLM 生成）
- `data/`（gitignored）dump、parquet、sqlite；`tmp/`（gitignored）官方 notebook 参考件

## 扩展层纪律（记忆 / 检索增强）

- 默认路径（NullMemory + 基础 WikiEnv）必须与 baseline 逐字节一致，回归通过后才允许接线
- 记忆库只允许从调参集（off0）trace 挖掘并冻结；held-out（off500）评测前不得更新
- 每个新能力单独跑同一 seed-233 500 题，作为消融表独立一行；任何扩展不得并入 baseline

## 测量噪声底线（重要，解释所有消融前先读这条）

- DeepSeek temp=0 **不保证逐位确定**：同 prompt 复跑，500 题中 84 题（17%）轨迹分叉、42 题（8.4%）预测改变，净 EM 摆动实测达 ±1.6pp（selfcheck 臂与 baseline 仅差 2 次否决，EM 却 47.2→45.6）
- 判读规则：|ΔEM| < ~2pp 一律视为噪声（McNemar p>0.05 佐证）。三轮消融下来，真实效应只有 4 个：rag -4.6pp、memory-combo -5.0pp（显著负）；cotsc +5.6pp、content +3.2pp、cotsc+content +7.2pp（显著正，见下）
- 第二轮消融实测：fallback 47.0%（重试 489 次仅救回 13 题，p=1.0）、concise 48.4%（+1.2 n.s.，近似错 34→28）、selfcheck 45.6%（否决仅触发 2 次，额外 +412 次验证调用，降分全是抖动）
- 第三轮消融实测（McNemar vs baseline 47.2%）：**cotsc 52.8%（+5.6pp，p=6e-5，成本仅 +$0.3——只在 25% 强制结束题触发）；ReAct-SC 52.4%（+5.2pp，p=7e-4，但 5× 轨迹成本 $11.3——ROI 远低于 cotsc）；content 50.4%（+3.2pp，p=0.052 边缘显著，ff 25%→18%，成本反而降 $0.13）；fallback2 46.4%（-0.8 噪声）**
- 离线门槛拦截记录：selfcheck-strict（错答误放 15/20、对答误杀 2/10，不上 500）；fallback2 初版（FTS OR 匹配未排序产生错页，加全 token 过滤后仍是噪声）；content 离线 542/599 救回、24% 页面含 gold，过门槛
- **最终系统 = cotsc+content 组合臂：EM 54.4%（+7.2pp，p=1.4e-5）、F1 70.7%、ff 18.0%、成本 $2.14 与 baseline 持平**。两臂正交叠加（cotsc 救步数耗尽、content 救检索失败）；增量检验：content 在 cotsc2 之上 +1.6pp（n.s.），cotsc 在 content 之上 +4.0pp（p=0.005）。文件 tag 为 cotsccontent（旧 combo tag 属第一轮 memory 组合，勿混淆）
- **held-out（off500，从未调参）验证通过：baseline 45.8% → combo 51.6%（+5.8pp，p=2e-4）**——调参集 +7.2pp 在未见数据上复现为 +5.8pp，轻微衰减但强显著，改进非过拟合
- 四阶段失败归因精读（各 50 条，存 `results/failure_attribution.json`）：残桶 A 粒度 42%（归一化层结构性问题）+ B 检索未达 26%（多为题干拼写错/前提错/页面缺失，不可救）+ F 标签噪声 ~10%（EM 天花板 ~94-95%）

## 常用命令（在项目根目录执行）

```bash
# 建索引（先解包 dump：tar -xjf data/enwiki-abstracts.tar.bz2 -C data/wiki_abstracts）
.venv/Scripts/python scripts/build_wiki_index.py --src data/wiki_abstracts --out data/wiki_abstracts.db
# 内容级索引（content/组合臂用）
.venv/Scripts/python scripts/build_content_index.py
# 跑评测（method: direct | cot | react | actonly；默认 --wiki-db data/wiki_full.db）
.venv/Scripts/python src/run_eval.py --method react --n 20 --workers 4
# 最终系统（cotsc+content）与 held-out
.venv/Scripts/python src/run_eval.py --method react --n 500 --workers 16 --cotsc --env content --tag combo
.venv/Scripts/python src/run_eval.py --method react --n 500 --offset 500 --workers 16 --cotsc --env content --tag heldout
```
