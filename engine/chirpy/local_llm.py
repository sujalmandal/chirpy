"""Bundled tiny LLM so Chirpy answers out of the box.

Runs ``Qwen/Qwen2.5-0.5B-Instruct`` directly inside the agent worker using Hugging
Face ``transformers`` on Apple Silicon (MPS) — no separate server. It's exposed
as a LiveKit ``llm.LLM`` so the ``AgentSession`` consumes it exactly like any
other provider.

The model is small by design; it trades some answer quality for a fully
offline, zero-config default. If the checkpoint isn't downloaded yet, or
inference fails, the stream returns a short, friendly message telling the user
the local model needs to finish downloading rather than staying silent.
"""

from __future__ import annotations

import asyncio
import logging
import os

from livekit.agents import APIConnectOptions
from livekit.agents import llm as llm_mod

logger = logging.getLogger("chirpy.local_llm")

DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

# Generation tuned for short, spoken replies.
MAX_NEW_TOKENS = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "220"))
TEMPERATURE = float(os.environ.get("LOCAL_LLM_TEMPERATURE", "0.7"))
TOP_P = 0.9

_NOT_READY = (
    "My local model is still finishing its first-time download. "
    "Give it a moment and try again in a few seconds."
)


def _mps() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _build_prompt(chat_ctx: llm_mod.ChatContext) -> str:
    """Render the session's chat context into Qwen's ChatML prompt."""
    msgs = chat_ctx.messages() if callable(chat_ctx.messages) else chat_ctx.messages
    parts: list[str] = []
    for msg in msgs:
        role = getattr(msg, "role", None)
        text = getattr(msg, "text_content", None) or getattr(msg, "raw_text_content", None) or ""
        if not text:
            continue
        role_name = str(role)
        # LiveKit uses "user"/"assistant"; Qwen ChatML expects the same tokens.
        parts.append(f"<|im_start|>{role_name}\n{text}<|im_end|>")
    if not parts:
        parts.append("<|im_start|>system\nYou are a concise voice assistant.<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class _LocalQwenStream(llm_mod.LLMStream):
    def __init__(self, local: "LocalQwenLLM", *, chat_ctx, conn_options):
        super().__init__(llm=local, chat_ctx=chat_ctx, tools=[], conn_options=conn_options)
        self._local = local

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._local._generate, self._chat_ctx)
        except Exception:
            logger.exception("local Qwen generation failed")
            text = _NOT_READY
        if not text:
            text = _NOT_READY
        self._event_ch.send_nowait(
            llm_mod.ChatChunk(
                id="local-qwen",
                delta=llm_mod.ChoiceDelta(role="assistant", content=text),
            )
        )


class LocalQwenLLM(llm_mod.LLM):
    """A LiveKit LLM that runs Qwen2.5-0.5B in-process via transformers."""

    def __init__(self, repo_id: str = DEFAULT_REPO, device: str | None = None):
        super().__init__()
        self._repo_id = repo_id
        self._device = device or ("mps" if _mps() else "cpu")
        self._model = None
        self._tok = None

    @property
    def model(self) -> str:
        return self._repo_id

    @property
    def provider(self) -> str:
        return "local-qwen"

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self._repo_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._repo_id, torch_dtype=torch.float16
        ).to(self._device)
        self._model.eval()
        logger.info("local Qwen loaded: %s on %s", self._repo_id, self._device)

    def _generate(self, chat_ctx: llm_mod.ChatContext) -> str:
        self._ensure()
        import torch

        prompt = _build_prompt(chat_ctx)
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=self._tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    def chat(
        self,
        *,
        chat_ctx,
        tools=None,
        conn_options: APIConnectOptions | None = None,
        **kwargs,
    ) -> llm_mod.LLMStream:
        conn_options = conn_options or APIConnectOptions()
        return _LocalQwenStream(self, chat_ctx=chat_ctx, conn_options=conn_options)
