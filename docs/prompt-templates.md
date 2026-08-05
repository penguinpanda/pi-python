# 提示模板（Python 移植）

提示模板是 Markdown 片段，输入 `/name` 展开为完整提示。加载与参数解析在 `src/pi_coding_agent/prompt_templates.py`，参数替换委托给 `src/pi_agent/prompt_templates.py` 的 `substitute_args`。

## 位置

- 全局：`~/.pi/agent/prompts/*.md`（source=`user`）
- 项目：`.pi/prompts/*.md`（项目信任后，source=`project`）
- settings 的 `prompts` 数组：文件或目录（source=`path`）
- 运行时显式加载：`PromptTemplateLoader.load(explicit_paths=[...])`

CLI 暂未提供 `--prompt-template` / `--no-prompt-templates` 标志（与 TS 不同）。

## 格式

````markdown
---
description: Review staged git changes
argument-hint: "<PATH>"
---
Review the staged changes (`git diff --cached`). Focus on:
- Bugs and logic errors
- Security issues
- Error handling gaps
````

- 文件名（不含 `.md`）即命令名：`review.md` → `/review`。
- `description` 缺省时取正文第一个非空行；超过 60 字符截断并追加 `...`。
- `argument-hint` 可选，用于自动补全里提示参数。
- 加载是**非递归**的：`prompts/` 只扫描一层，子目录里的模板需要通过 settings / 显式路径加入。

## 参数

支持 TS 同款替换语法（`src/pi_agent/prompt_templates.py`）：

- `$1`、`$2` ... 位置参数
- `$@` / `$ARGUMENTS` 全部参数
- `${1:-default}` / `${@:-default}` / `${ARGUMENTS:-default}` 带默认值
- `${@:N}` 从第 N 个参数开始；`${@:N:L}` 取 L 个

`parse_command_args` 支持单双引号分组的参数解析。

## 展开

`_session.expand_prompt` 先尝试 `/skill:name`，再尝试 `/templateName [args]`；未匹配时原样返回（`src/pi_coding_agent/_session.py`）。
