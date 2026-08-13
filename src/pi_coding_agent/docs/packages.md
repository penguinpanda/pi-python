# Pi 包（Python 移植状态）

TS 的 `pi install / remove / update / config` 已移植核心子集：支持 npm / git / 本地包源，写入全局或项目 `packages` 配置。`pi config` 现在会打开交互式资源配置选择器，用于启用或禁用扩展 / 技能 / 提示模板 / 主题。

## 现状：资源如何加载

pi-python 通过 `ResourceLoader` 从以下位置加载扩展 / 技能 / 提示模板 / 主题（`src/pi_coding_agent/resource_loader.py`）：

- 全局：`~/.pi/agent/{extensions,skills,prompts,themes}/`
- 项目（信任后）：`.pi/{extensions,skills,prompts,themes}/`
- settings 中的显式路径（如 `skills`、`prompts` 数组）
- CLI 覆盖：`--system-prompt` / `--append-system-prompt` / `--tools` / `--exclude-tools` / `--no-tools` 等

对应的加载器：

- 扩展：`src/pi_coding_agent/extensions/`
- 技能：`src/pi_coding_agent/skills.py`（`SkillLoader`）
- 提示模板：`src/pi_coding_agent/prompt_templates.py`（`PromptTemplateLoader`）
- 主题：`src/pi_tui/theme.py`

## 与 TS 的差异

- 无 `package.json` 的 `pi` manifest（`extensions` / `skills` / `prompts` / `themes` 键）。
- 无安装后自动补齐依赖（npm install）流程。
- 无全局 / 项目同名包覆盖和 `autoload: false` 增量语义。

## 参考

- 资源加载：`src/pi_coding_agent/resource_loader.py`
- 设置项：`src/pi_coding_agent/settings_manager.py`
- TS 原文档：TS 仓库 `packages/coding-agent/docs/packages.md`
