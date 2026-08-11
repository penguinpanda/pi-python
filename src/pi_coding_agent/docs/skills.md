# 技能（Python 移植）

技能是带 frontmatter 的 `SKILL.md` 能力包，按需由模型加载。实现：`src/pi_coding_agent/skills.py`（发现 / 校验 / 加载）、`src/pi_agent/skills.py`（底层校验与格式化）、`pi_coding_agent.skills.format_skills_for_prompt`（系统提示 XML）。

## 位置

- 全局：`~/.pi/agent/skills/`
- 项目：`.pi/skills/`（项目信任后）
- settings 的 `skills` 数组 / `SkillLoader.load(explicit_paths=[...])` 显式路径

`~/.agents/skills` 和祖先目录 `.agents/skills` **尚未移植**（TS 有）。CLI 暂未提供 `--skill` / `--no-skills` 标志。

## 发现规则

- 目录含 `SKILL.md` → 视为一个技能，不再递归。
- 全局 / 项目根目录下的 `.md` 文件也作为单个技能加载（`include_root_files=True`）。
- 跳过 gitignore 模式（`.gitignore` / `.ignore` / `.fdignore`）、`.` 开头目录和 `node_modules`。

## 格式与校验

- `name`：1-64 字符，`[a-z0-9-]`，不能以 `-` 开头 / 结尾、不能含连续 `--`；缺省时取父目录名。校验失败只是 warning，技能仍加载（对齐 TS 的宽松策略）。
- `description`：必填，超过 1024 字符给 warning；**缺 description 的技能不加载**。
- `disable-model-invocation: true`：不进系统提示，只能通过 `/skill:name` 手动调用。
- 同名冲突：保留先加载的，记录 `collision` diagnostic。

## 系统提示

`format_skills_for_prompt` 输出：

```xml
<available_skills>
  <skill>
    <name>...</name>
    <description>...</description>
    <location>/abs/path/SKILL.md</location>
  </skill>
</available_skills>
```

并提示模型：任务匹配 description 时用 read 加载 SKILL.md；相对路径按技能目录（SKILL.md 所在目录）解析。

## 调用

`/skill:name [instructions]` 展开为 `<skill name=... location=...>` 块，参数追加在块后（`src/pi_coding_agent/_session.py` 的 `_expand_skill_command`）。可用 `enableSkillCommands` 设置开关（`src/pi_coding_agent/settings_manager.py`，默认开启）。
