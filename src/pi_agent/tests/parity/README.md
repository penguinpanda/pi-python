# System Prompt Runtime Parity Tests

验证 `pi_coding_agent.system_prompt.build_system_prompt` 与 TypeScript 侧
`packages/coding-agent/src/core/system-prompt.ts` 的 `buildSystemPrompt` 在
**运行时输出**上逐字符一致（behavior parity，不是源码模板一致性）。

## 目录结构

```
src/pi_agent/tests/parity/
├── fixtures/                  # 固定 options 输入（camelCase，对齐 TS BuildSystemPromptOptions）
│   ├── default.json           #   默认工具 + snippets + guidelines + append + context + skills
│   ├── custom_prompt.json     #   customPrompt 分支（含 XML 特殊字符 skill）
│   ├── no_read.json           #   仅有 bash：无 read → 无 skills 段 + bash 探索指南
│   ├── null_tools.json        #   selectedTools: null → 默认 4 工具回退
│   ├── empty_tools.json       #   selectedTools: [] → "Available tools:\n(none)"
│   ├── no_skills.json         #   skills: [] → 无 skills 段
│   └── disabled_skill.json    #   disableModelInvocation=true 的技能被过滤
├── golden/                    # TS 侧真实输出（在 pi TS mono-repo 中生成后拷入，已入库）
├── python_out/                # Python 侧输出（由 dump_system_prompt.py 生成，不入库）
├── dump_system_prompt.py      # Python dump：fixture → python_out/<name>.txt
├── compare_outputs.py         # 一键：Python dump → 与 golden 逐字符比较
├── test_system_prompt_parity.py
├── test_compaction_prompts_parity.py  # compaction 常量：Python 运行时值 vs TS golden
├── test_compaction_functions_parity.py # compaction 函数：固定输入 vs TS 真实函数输出
└── test_prompt_templates_parity.py    # 模板展开：substitute/parseCommandArgs/expand
```

## 一键比较（日常使用）

运行 `compare_outputs.py` 会先执行 Python 侧 dump 再与已入库的 golden 比较，
修改 Python 提示词后跑一次即可立刻看到与 TS golden 的差异；有差异时退出码为 1：

```bash
uv run --no-sync python src/pi_agent/tests/parity/compare_outputs.py
```

流程：`python dump_system_prompt.py`（刷新 python_out/）→ 与 golden/ 逐文件
unified diff。

## Python 侧 dump

```bash
uv run --no-sync python src/pi_agent/tests/parity/dump_system_prompt.py
```

纯 dump，不比较。

## pytest（golden 已入库，直接跑）

```bash
uv run --no-sync pytest src/pi_agent/tests/parity -q
```

## 重新生成 golden（TS 侧变更后）

golden 由 pi TS mono-repo 中的 dump 脚本（system prompt / compaction
常量 / compaction 函数 / prompt 模板函数四组）生成，脚本不随本仓库分发。
TS 侧变更后需在装有 pi mono-repo 依赖的机器上运行对应 dump 脚本，把
`golden/*.txt` 拷回本目录入库；`PI_PACKAGE_DIR` 必须固定为 `C:/pi-pkg`
（与测试内 `PACKAGE_DIR` 一致）。golden 缺失时 pytest 会 skip，不会失败。

## 添加新 fixture

1. 在 `fixtures/` 新增 `<name>.json`（字段名用 camelCase，对齐 TS options）。
2. 在 pi TS mono-repo 中生成对应的 `golden/<name>.txt` 并拷入，运行
   `compare_outputs.py` 确认 Python 输出与 golden 一致。
3. 测试自动参数化覆盖新 fixture。

## 平台说明

golden 中的路径统一为正斜杠形式（如 `C:/pi-pkg/README.md`、cwd 的
`C:/coding/test-project`）。`build_system_prompt` 输出前会把 cwd 与 pi 包
文档路径的分隔符统一为 `/`，`get_package_dir()` 对 `PI_PACKAGE_DIR` 不做
resolve（避免 Windows 形式路径在 POSIX 上被当作相对路径拼上 cwd），因此
Windows 与 POSIX 上输出逐字符一致，parity 断言可跨平台运行。
