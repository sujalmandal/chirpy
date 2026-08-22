"""LLM fallback so the agent replies helpfully when the LLM is unconfigured.

When the user hasn't set a valid OpenAI-compatible LLM endpoint/model (Settings
is blank or points at a server that isn't running), the normal LLM request
fails and the agent would otherwise stay silent. This module wraps the
configured LLM: if there is none, or if a chat request errors, it returns a
short spoken reply telling the user to configure their LLM instead of failing
silently.

The wrapper is a drop-in ``livekit.agents.llm.LLM`` subclass used by
:func:`agent.build_session` in place of the raw ``openai.LLM``.
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import APIConnectOptions
from livekit.agents import llm as llm_mod

logger = logging.getLogger("chirpy.llm_fallback")

# Short, spoken-friendly guidance. Kept concise so it reads naturally when
# synthesized and doesn't bloat the conversation view.
CONFIG_GUIDANCE = (
    "Please configure your language model. Open the debug window, go to Settings, "
    "and enter a valid endpoint, model, and API key. Then press Save and Restart."
)

# Fail fast on the inner LLM so the guidance appears quickly instead of after a
# long retry/backoff loop against a dead endpoint.
_FAST_FAIL = APIConnectOptions(max_retry=0)


class _FallbackStream(llm_mod.LLMStream):
    def __init__(self, fallback, *, chat_ctx, conn_options, inner_stream=None):
        super().__init__(
            llm=fallback,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=conn_options,
        )
        self._inner = inner_stream

    async def _run(self) -> None:
        # A configured LLM succeeded in producing a stream: forward its chunks.
        if self._inner is not None:
            try:
                async for chunk in self._inner:
                    self._event_ch.send_nowait(chunk)
                return
            except Exception:
                # The configured LLM errored (bad endpoint, bad model, ...).
                logger.warning("LLM chat failed; replying with config guidance", exc_info=True)

        self._event_ch.send_nowait(
            llm_mod.ChatChunk(
                id="configure-llm",
                delta=llm_mod.ChoiceDelta(role="assistant", content=CONFIG_GUIDANCE),
            )
        )


class FallbackLLM(llm_mod.LLM):
    """Wrap an optional inner LLM; on absence or error, emit config guidance."""

    def __init__(self, inner: llm_mod.LLM | None):
        super().__init__()
        self._inner = inner

    @property
    def model(self) -> str:
        return self._inner.model if self._inner is not None else "unconfigured"

    @property
    def provider(self) -> str:
        return "chirpy-fallback"

    def prewarm(self, *, loop=None) -> None:
        if self._inner is not None:
            self._inner.prewarm(loop=loop)

    def chat(
        self,
        *,
        chat_ctx,
        tools=None,
        conn_options: APIConnectOptions | None = None,
        **kwargs: Any,
    ) -> llm_mod.LLMStream:
        conn_options = conn_options or APIConnectOptions()
        inner_stream = None
        if self._inner is not None:
            try:
                # Use max_retry=0 so a dead/bad endpoint raises on the first
                # attempt and we can reply with guidance quickly.
                inner_stream = self._inner.chat(
                    chat_ctx=chat_ctx,
                    tools=tools or [],
                    conn_options=_FAST_FAIL,
                    **kwargs,
                )
            except Exception:
                logger.warning("could not start LLM chat; using config guidance", exc_info=True)
                inner_stream = None
        return _FallbackStream(
            self,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            inner_stream=inner_stream,
        )

    async def aclose(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()
        await super().aclose()
