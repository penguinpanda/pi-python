# syntax=docker/dockerfile:1
#
# pi-python coding-agent 隔离运行镜像。
# 构建：docker compose -f docker/compose.yaml build
#
# 设计原则：
#   - 非 root 用户（pi，uid/gid 1000）
#   - 只拷贝 src 与依赖清单，不打包宿主任何真实数据
#   - 运行时代码位于 /app（只读根文件系统下由 compose 挂为只读）
#   - 工作目录 /workspace 由 compose 绑定到一次性目录

FROM ghcr.io/astral-sh/uv:0.12.0 AS uv-image

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

# 非 root 用户 pi
RUN groupadd --gid 1000 pi \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash pi \
    && mkdir -p /app/src /workspace \
    && chown -R pi:pi /app /workspace /home/pi

COPY --from=uv-image /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# 只安装运行时依赖，不安装项目本身（PYTHONPATH 指向 src）
RUN uv sync --no-dev --no-install-project \
    && rm -rf /root/.cache /tmp/*

USER pi
WORKDIR /workspace
ENTRYPOINT ["/app/.venv/bin/python", "-m", "pi_coding_agent"]
