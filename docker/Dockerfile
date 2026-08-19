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

# 构建期 PyPI 镜像源。uv.lock 内的 artifact 与 registry 均为官方 PyPI；
# 默认与锁文件一致，避免 uv 因索引不匹配重新解析（TUNA 对容器/uv 请求可能返回 403）。
# 网络可达的自建镜像可用 --build-arg PYPI_INDEX_URL=... 覆盖。
ARG PYPI_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${PYPI_INDEX_URL} \
    UV_HTTP_TIMEOUT=60 \
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

# 容器内提供 `pi-python` 命令（项目未安装时也生效）。
RUN printf '#!/bin/sh\nexec /app/.venv/bin/python -m pi_coding_agent "$@"\n' \
        > /usr/local/bin/pi-python \
    && chmod +x /usr/local/bin/pi-python

# pi-evals 评测入口（等价于宿主 `uv run pi-evals`）。
RUN printf '#!/bin/sh\nexec /app/.venv/bin/python -m pi_evals "$@"\n' \
        > /usr/local/bin/pi-evals \
    && chmod +x /usr/local/bin/pi-evals

USER pi
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/pi-python"]
