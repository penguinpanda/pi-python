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
├── golden/                    # TS 侧真实输出（system prompt + compaction 常量，已入库）
├── python_out/                # Python 侧输出（由 dump_system_prompt.py 生成，不入库）
├── dump-system-prompt.ts      # TS dump：fixture → golden/<name>.txt
├── dump-compaction-prompts.ts # TS dump：compaction 模板常量 → golden/compaction_*.txt
├── dump-compaction-functions.ts # TS dump：真实 formatFileOperations/serializeConversation → golden
├── dump_system_prompt.py      # Python dump：fixture → python_out/<name>.txt
├── compare_outputs.py         # 一键：TS dump → Python dump → 逐字符比较
├── test_system_prompt_parity.py
├── test_compaction_prompts_parity.py  # compaction 常量：Python 运行时值 vs TS golden
├── test_compaction_functions_parity.py # compaction 函数：固定输入 vs TS 真实函数输出
└── test_prompt_templates_parity.py    # 模板展开：substitute/parseCommandArgs/expand
```

## 一键比较（日常使用）

运行 `compare_outputs.py` 会依次执行两侧 dump 再比较，修改提示词（TS 或
Python）后跑一次即可立刻看到差异；有差异时退出码为 1：

```bash
uv run --no-sync python src/pi_agent/tests/parity/compare_outputs.py
```

流程：`node dump-system-prompt.ts`（刷新 golden/）→ `python
dump_system_prompt.py`（刷新 python_out/）→ 逐文件 unified diff。

## 单侧 dump

- TS 侧：`PI_PACKAGE_DIR=C:/pi-pkg node --experimental-strip-types src/pi_agent/tests/parity/dump-system-prompt.ts`
- Python 侧：`uv run --no-sync python src/pi_agent/tests/parity/dump_system_prompt.py`

两侧 dump 均为纯 dump，不比较。

## pytest（golden 已入库，直接跑）

```bash
uv run --no-sync pytest src/pi_agent/tests/parity -q
```

## 重新生成 golden（TS 侧变更后）

### System prompt（fixture 输出）

需要 pi mono-repo 依赖已安装（pi 仓库根 `npm install`，运行时只需
`ignore`、`yaml`、`cross-spawn`）与 Node >= 22.6。

- `PI_PACKAGE_DIR` 必须固定为 `C:/pi-pkg`（与测试内 `PACKAGE_DIR` 一致），
  否则 "Pi documentation" 段的路径两边不同。
- 可通过 `PI_TS_SYSTEM_PROMPT_SRC` 覆盖 TS 源码路径（默认
  `C:/coding/AI/Agent/pi/packages/coding-agent/src/core/system-prompt.ts`）。
- golden 文件缺失时 pytest 会 skip 并提示上述命令，不会失败。

### Compaction / branch-summary 模板常量

```bash
node --experimental-strip-types src/pi_agent/tests/parity/dump-compaction-prompts.ts
```

从 TS 源码**原样提取**（不做 normalize/strip）`SUMMARIZATION_SYSTEM_PROMPT`、
`SUMMARIZATION_PROMPT`、`UPDATE_SUMMARIZATION_PROMPT`、
`TURN_PREFIX_SUMMARIZATION_PROMPT`、`BRANCH_SUMMARY_PROMPT`、
`BRANCH_SUMMARY_PREAMBLE` 六个常量到 `golden/compaction_*.txt`。这些常量未从
TS 模块导出（消费它们的函数会触发 LLM 调用），但模板字面量本身即运行时值。
Python 侧由 `test_compaction_prompts_parity.py` 用运行时值（import）逐字符比对。
可用 `PI_TS_COMPACTION_DIR` 覆盖 TS 源码目录。

### Compaction 函数（formatFileOperations / serializeConversation）

```bash
node --experimental-strip-types src/pi_agent/tests/parity/dump-compaction-functions.ts
```

运行真实 TS 函数（fixtures/compaction_functions.json 的固定输入 →
`golden/compaction_formatFileOperations_<i>.txt`、`compaction_serializeConversation_<i>.txt`）。
前置：TS 侧 `@earendil-works/pi-ai` 无构建产物 dist/，需要在 pi 仓库
`node_modules/@earendil-works/pi-ai/` 放一个最小 shim（package.json + index.js
re-export `packages/ai/src/utils/text.ts` 的 contentText；node_modules 被
.gitignore，不污染仓库）。Node 22 禁止对 node_modules 下的 .ts 做 type
stripping，所以 shim 必须是 .js。

2025 年首次建立时该测试发现并修复了一个真实差异：Python `json.dumps` 默认
分隔符带空格（`{"x": 1}`），TS `JSON.stringify` 无空格（`{"x":1}`），已在
`pi_coding_agent/compaction.py` 用 `separators=(',', ':')` 对齐。

### Prompt 模板函数（substituteArgs / parseCommandArgs / expandPromptTemplate）

```bash
node --experimental-strip-types src/pi_agent/tests/parity/dump-prompt-templates.ts
```

运行真实 TS 函数（`packages/coding-agent/src/core/prompt-templates.ts`，
fixtures/prompt_templates.json → `golden/prompttemplates_*.txt`）。Python 侧
对齐 coding-agent 层（`pi_agent.prompt_templates.substitute_args`、
`pi_coding_agent.prompt_templates.parse_command_args` 与
`PromptTemplateLoader.expand`）。注意 TS agent 层（`packages/agent`）的
`substituteArgs` 是旧版（无 `${N:-default}` 默认值语法），Python 对齐的是
coding-agent 层的新版。当前 24 个用例全部一致。

## 添加新 fixture

1. 在 `fixtures/` 新增 `<name>.json`（字段名用 camelCase，对齐 TS options）。
2. 运行 `compare_outputs.py` 重新生成两侧输出并确认一致，把 `golden/<name>.txt` 入库。
3. 测试自动参数化覆盖新 fixture。

## 平台说明

golden 中的路径基于 Windows 形式（如 `C:\pi-pkg\README.md`、cwd 的
`C:/coding/test-project`）。parity 断言在 Windows 上严格逐字符；在其他平台
路径分隔符可能不同，需要重新生成 golden。
