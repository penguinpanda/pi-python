# 开发规则（pi-python）

## 对话风格

- 回答先于修改：用户提问时先回答，再做编辑或执行实现命令。
- 保持简短直接，技术性文字；不用表情符号。
- 回应反馈 / 评审时，先明确同意或不同意，再说明改了什么。

## 代码质量

- 做大范围改动、修改未完整读过的文件、或做审计时，先完整读文件，不要只依赖搜索片段。
- 避免 `Any`；确有必要时加注释说明原因。
- Python 3.10+：优先 `dataclass`、`pathlib`、类型注解；源文件统一加
  `from __future__ import annotations`。
- 顶层导入优先；仅在循环依赖 / 延迟加载场景允许局部导入，并加注释。
- 不删除看起来有意的功能；用户没有要求时不做向后兼容。
- 键盘 / 快捷键等可配置项写进 `DEFAULT_*` 默认值，不硬编码按键。
- 不直接手改生成物（如模型元数据生成文件）；改生成脚本后重新生成。

## 命令

- 代码改动（非纯文档）后运行 `python scripts/check.py`
  （ruff lint + ruff format + mypy + pytest），看完整输出并修复全部问题。
- 快速迭代：`pytest -q src/pi_coding_agent/tests/test_x.py -k ...`；
  新增或修改测试后必须跑通对应用例。
- 测试一律用 faux provider，禁止真实 API key / 付费 token。
- 临时脚本写到临时文件再运行，用完删除；不要把多行脚本直接嵌进 shell 命令。
- 用户没有要求时不要提交（commit）。

## 依赖与安装

- 依赖 / `uv.lock` 变更视为已评审代码；新增依赖用 `uv add` / `uv add --dev`。
- 不静默绕过 pre-commit；不用 `--no-verify` 提交。

## Git

- 只提交本次会话自己改的文件；用显式路径暂存，禁止 `git add -A` / `git add .`。
- 提交前 `git status` 确认暂存内容只包含自己的文件。
- 提交信息格式：`{feat,fix,docs,test,chore}[(ai,agent,coding-agent,tui)]: 摘要`。
- 禁止 `git reset --hard`、`git checkout .`、`git clean -fd`、`git stash`、
  `git commit --no-verify`。
- 多会话共享工作区时：git 操作只碰自己的文件；rebase 冲突只解决自己改过的文件。

## 结构约定

- src 布局：`pi_ai` / `pi_agent` / `pi_coding_agent` / `pi_tui` / `pi_protocol` /
  `pi_storage` / `pi_server` / `pi_evals`。
- `pi_tui` 是独立可复用 TUI 框架；overlay 运行时在 `src/pi_tui/overlay/`，
  核心（model/layout/focus/manager）不依赖 Textual，便于单测。
- 文档：`docs/tui.md`（用户文档）、`docs/tui-gap.md`（TS 差距与路线图）、
  `examples/extensions/`（与 TS 同步的示例，状态见 `STATUS.md`）。
- 测试：`src/pi_tui/tests/`（框架层）、`src/pi_coding_agent/tests/`（应用层）。
- `CHANGELOG.md` 只在 `[Unreleased]` 下追加；已发布版本段不可修改。
