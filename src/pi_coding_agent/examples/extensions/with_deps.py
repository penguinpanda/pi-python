"""Extension with local dependencies - import helper from a sibling package.

Python port of with-deps/（依赖解析：扩展目录已加入 sys.path）。
依赖放在 with_deps_lib/（子目录不会被当作扩展加载）。
"""

from with_deps_lib.helper import greet

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def handler(ctx, args: str) -> None:
        ctx.ui.notify(greet(args or "pi"), "info")

    pi.register_command(
        "with-deps",
        {
            "description": "Greet using a local helper dependency",
            "handler": handler,
        },
    )
