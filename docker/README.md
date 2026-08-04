# pi-coding-agent 隔离运行环境（Docker）

在一次性容器里运行 pi coding-agent，把 agent 的 write/edit/bash 等工具的
破坏范围限制在容器内。**这是文件系统与进程隔离，不是网络隔离**：模型必须联网
才能调用 LLM API，所以不要往容器里放任何真实数据。

## 安全边界

- 容器内是 Linux，`find /`、`rm -rf` 等命令只会作用于容器自身的文件系统，碰不到宿主机。
- 只挂载一次性目录 `work/temp` 作为工作区；不要挂载 `~/.ssh`、`~/.pi`、真实代码库。
- API key 只通过环境变量注入（`docker/.env` 或宿主环境变量），不写入镜像。
- 会话与认证数据写入命名卷 `pi-home`（隔离在 Docker 卷内，不在宿主机文件系统），
  保留到 `docker compose down -v` 才清空。
- 容器以非 root 用户运行，丢弃全部 Linux capabilities，限制内存 2G / 2 核，根文件系统只读。

## 前置条件

- Docker Desktop（Windows，需启用 WSL2 后端）或 Docker Engine（Ubuntu）。
- 构建时需要网络（拉取基础镜像与 Python 依赖）。

## 快速开始

```powershell
# 1. 配置 API key（二选一）
Copy-Item docker\.env.example docker\.env   # 然后编辑填入 key
# 或直接设置宿主环境变量，如 $env:DEEPSEEK_API_KEY = "sk-..."

# 2. 构建镜像（首次较慢）
docker compose -f docker/compose.yaml build

# 3. 运行（print 模式）
.\docker\run.ps1 -p "read notes.md and summarize"
```

Linux/macOS 对应：

```bash
cp docker/.env.example docker/.env   # 编辑填入 key
docker compose -f docker/compose.yaml build
./docker/run.sh -p "read notes.md and summarize"
```

其他模式：

```bash
# TUI（需要真实终端）
docker compose -f docker/compose.yaml run --rm pi --mode tui

# RPC（stdin/stdout JSONL，-T 关闭 TTY）
echo '{"type":"get_state","id":"1"}' |
  docker compose -f docker/compose.yaml run --rm -T pi --mode rpc

# 离线验证 Faux provider
docker compose -f docker/compose.yaml run --rm pi --provider faux --model faux-1 -p "hi"
```

## 进入容器内连续测试（可选）

两个服务共用同一镜像，构建一次即可：

- `pi`：一次性命令（`docker compose run --rm pi ...`），跑完即删。
- `pi-dev`：**常驻容器**（推荐长时间测试），`up -d` 启动后一直存活，
  用 `docker compose exec` 反复执行测试；会话/认证持久化在命名卷 `pi-home`。

### 常驻容器（pi-dev）

```powershell
# 1) 启动常驻容器（需要先 build 出镜像）
docker compose -f docker/compose.yaml up -d pi-dev

# 2) 反复执行测试（每条命令都进入同一个容器）
docker compose exec pi-dev python -m pi_coding_agent -p "read notes.md and summarize"
docker compose exec pi-dev python -m pi_coding_agent --mode tui   # TUI 需真实终端

# 3) 查看会话历史（命名卷，跨命令保留）
docker compose exec pi-dev ls -lt /home/pi/.pi/agent/sessions/

# 4) 停止；彻底清理（含命名卷里的会话）用 down -v
docker compose -f docker/compose.yaml stop pi-dev
docker compose -f docker/compose.yaml down -v
```

进入交互 shell 连续操作：

```bash
docker compose exec pi-dev /bin/bash
export PATH=/app/.venv/bin:$PATH
python -m pi_coding_agent --version
python -m pi_coding_agent -p "read notes.md and summarize"
exit
```

一次性容器也可以进 shell（不常驻）：`docker compose run --rm --entrypoint bash pi`。

### 一键重建并进入容器

每次改完源码想直接进容器测试，用脚本（自动 build + `--force-recreate` + 进入 shell，
不用再手动记命令）：

```powershell
.\docker\dev-enter.ps1
```

Linux/macOS：

```bash
./docker/dev-enter.sh
```

脚本内部执行：`docker compose build` → `up -d --force-recreate pi-dev` →
`docker compose exec pi-dev bash`。容器名不写死，按服务名 `pi-dev` 进入。
不想每次都重建镜像的话，删掉脚本里的 `build` 行即可。

注意：容器内**不要用 `uv run`**——镜像没有宿主那样的项目环境，`uv run` 会新建
一个空环境，导致 `httpx` 等依赖缺失（ModuleNotFoundError）。直接使用 `python`
（即 `/app/.venv/bin/python`）。重建镜像后 `PATH` 已包含该目录，无需再 export。

## 使用纪律（每次实验）

- 只往 `work/temp` 放一次性测试文件；结束后清空或直接删除目录重建。
- 用完清理：`docker compose -f docker/compose.yaml down`。
- 定期重建镜像并更新依赖：`docker compose -f docker/compose.yaml build --pull --no-cache`。
- 会话持久化已默认启用（命名卷 `pi-home`），跨运行验证 C-28..C-33 直接可用；
  清空全部会话与认证用 `docker compose -f docker/compose.yaml down -v`。

## 进阶：网络白名单（可选）

默认容器可访问任意外网。如果要求更高，可只在网关放行 LLM API 域名
（如 `api.deepseek.com`、`api.openai.com`），其余流量全部拒绝，这样即使模型
拿到容器内数据也难以外传。实现方式（Linux 主机）：

```bash
docker network create pi-net
# 用 iptables 在容器网关限制出站目标域名（按 Docker 网络接口名调整）
```

Windows Docker Desktop 下做域名级出站限制较麻烦，建议在 Ubuntu 上做。

## 与宿主环境的差异

- 容器内 bash 工具使用 Linux bash，宿主 Windows 上"Git Bash 执行 `find /` 扫描全盘"
  的问题在容器内不存在（只扫容器自身文件系统）。
- Ollama：已接入。compose 默认注入 `OLLAMA_BASE_URL=http://host.docker.internal:11434`
  并启用 `extra_hosts`；前提是宿主机 Ollama 监听 `0.0.0.0`
  （Windows 设置环境变量 `OLLAMA_HOST=0.0.0.0` 后重启 Ollama）。
