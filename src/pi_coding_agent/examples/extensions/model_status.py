"""Model status extension - show model changes in the status bar.

Python port of model-status.ts. 使用 `model_select` 事件
（pi-python 已实现，事件带 model / previousModel 对象）。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def on_model_select(event, ctx):
        model = event.get("model")
        previous_model = event.get("previousModel")
        if model is None:
            return
        next_id = f"{model.provider}/{model.id}"
        prev = (
            f"{previous_model.provider}/{previous_model.id}"
            if previous_model is not None
            else "none"
        )
        ctx.ui.notify(f"Model: {next_id}", "info")
        ctx.ui.set_status("model", f"🤖 {model.id}")
        print(f"[model_select] {prev} → {next_id}")

    pi.on("model_select", on_model_select)
