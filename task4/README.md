# Task 4（RCM20）交接说明

本文件夹是交给组员合并的材料，**不是全组最终提交包**。本 README 仅供组内交接，不放入课程最终报告正文。

## 文件

| 文件 | 用途 |
| --- | --- |
| `CEG5302_Part2_Task3_Task4_RCM20_clean.ipynb` | Part 2 Notebook，已包含 Task 3 和 Task 4，并保留完整运行输出 |
| `task4_report_section.md` | 报告中 Task 4 一节的正文、Table 1、图注和参考文献 |
| `figures/author_parameter_effects.png` | 报告 Fig. 1 |
| `figures/author_final_front.png` | 报告 Fig. 2 |

## Notebook 怎么合并

Notebook 的结构：最前面是本文件的范围说明和一个独立运行用的 `import numpy as np` 单元格，然后从 `## Part 2. Constrained Multi-Objective Optimization` 开始是课程模板的 Part 2、队友的 Task 3（含唯一的 `CNSGA2` 类）和 Task 4。

**情况 A：主 Notebook 已经有 Task 3**

- 只复制从 `# Task 4 - RCM20 Hydro-static Thrust Bearing Design` 开始到结尾的所有单元格，放在 Task 3 之后。
- Task 4 的单元格自己导入了所需的库（numpy、matplotlib、json、hashlib、time、pathlib），只依赖 Task 3 已定义的 `CNSGA2` 类。
- **不要重复加入 `CNSGA2` 类。**

**情况 B：整体替换 Part 2**

- 从 `## Part 2. Constrained Multi-Objective Optimization` 开始复制到结尾，替换主 Notebook 中对应的 Part 2。
- 省略本 Notebook 最前面的范围说明和 `import numpy as np` 单元格。
- 确认主 Notebook 顶部已有 `import numpy as np`，模板 Part 2 的单元格需要它。

无论哪种情况，合并后的 Notebook 中都只能有一个 `class CNSGA2` 和一个 `class MW7`。

## 模型说明

Task 4 使用作者 MATLAB 源码（`CEC2021_func.m` 中 `func == 20` 分支）对应的模型，这是依据助教的建议。补充材料公式与源码的差异及选择理由已在报告中说明。

## 报告

`task4_report_section.md` 的正文和两张图交由负责报告的同学合并。章节号、图表号和参考文献编号请按全组报告统一调整。

## 运行

- 第三方库：`numpy`、`matplotlib`，以及运行 Notebook 所需的 `jupyter`（或 VS Code 的 Jupyter 扩展）和 `ipykernel`。
- 运行时会在 Notebook 所在目录生成：`task3_results/baseline-v1/`（Task 3 输出）、`results/`（Task 4 结果汇总）和 `figures/`（Task 4 图）。这些是运行产物，不放入提交 ZIP。
- 运行时间：Task 3 大约一分钟以内；Task 4 的完整参数实验在我们的机器上大约 20–40 分钟，实际取决于电脑性能和负载。
- 全组合并完成后，请 **Restart + Run All** 一次，确认从头到尾无报错。
