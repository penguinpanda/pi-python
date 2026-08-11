# Examples

pi-coding-agent SDK 与扩展示例。

## 目录

### [extensions/](extensions/)

扩展示例：

- 生命周期事件处理（工具拦截、安全门、上下文修改）
- 自定义工具（todo、提问、子代理、输出截断）
- 命令与键盘快捷键
- 自定义 UI（footer / header / editor / overlay）
- Git 集成（checkpoint、auto-commit）
- 系统提示修改与自定义压缩
- 外部集成（SSH、文件监听、系统主题同步）
- 自定义 provider（Anthropic 自定义流式、GitLab Duo）

`*.py` 为 Python 实现；`*.ts` 为 TS 原文（对齐 TS 仓库
`packages/coding-agent/examples/extensions/`），移植状态见
[extensions/STATUS.md](extensions/STATUS.md)。

## 文档

- [扩展文档](../docs/extensions.md)
- [技能文档](../docs/skills.md)
- [TUI 文档](../docs/tui.md)
