"""Voice engine — builds the AgentSession (STT/LLM/TTS) for ENGINE_PROFILE.

Same two engines as Rock Scheduler, inlined here so this folder stays
self-contained:
  - cascade  : Deepgram STT -> OpenAI LLM -> Cartesia TTS (plugins, or
               LiveKit Inference fallback when a vendor key is missing)
  - realtime : one OpenAI speech-to-speech model

This is the seam we will tune later for interruption + silence handling:
VAD sensitivity, endpointing delays, and turn detection all live here.
"""

from livekit.agents import (
    AgentSession,
    TurnHandlingOptions,
    inference,
    llm,
    stt,
    tts,
)
from livekit.plugins import cartesia, deepgram, openai, silero

import config


def _stt() -> stt.STT:
    if config.DEEPGRAM_API_KEY:
        return deepgram.STT(model="nova-3")
    return inference.STT(model="deepgram/nova-3", language="multi")


def _llm() -> llm.LLM:
    if config.OPENAI_API_KEY:
        return openai.LLM(model=config.LLM_MODEL)
    return inference.LLM(model=f"openai/{config.LLM_MODEL}")


def _tts() -> tts.TTS:
    if config.CARTESIA_API_KEY:
        return cartesia.TTS(model="sonic-3", voice=config.CARTESIA_VOICE)
    return inference.TTS(model=f"cartesia/sonic-3:{config.CARTESIA_VOICE}")


def _turn_detector() -> inference.TurnDetector:
    # unlikely_threshold=0 means "keep the model's per-language default".
    if config.EOT_UNLIKELY_THRESHOLD > 0:
        return inference.TurnDetector(unlikely_threshold=config.EOT_UNLIKELY_THRESHOLD)
    return inference.TurnDetector()


def _build_cascade() -> AgentSession:
    return AgentSession(
        stt=_stt(),
        llm=_llm(),
        tts=_tts(),
        vad=silero.VAD.load(),
        turn_handling=TurnHandlingOptions(
            turn_detection=_turn_detector(),
            # Cap how long an "unsure" pause waits (partial dict merges with
            # library defaults for the keys we omit).
            endpointing={
                "min_delay": config.EOT_MIN_DELAY,
                "max_delay": config.EOT_MAX_DELAY,
            },
            # Recover faster from false interruptions (default timeout 2.0s).
            interruption={
                "false_interruption_timeout": config.FALSE_INTERRUPTION_TIMEOUT,
            },
        ),
    )


def _build_realtime() -> AgentSession:
    return AgentSession(
        llm=openai.realtime.RealtimeModel(
            model=config.REALTIME_MODEL,
            voice=config.REALTIME_VOICE,
        ),
    )


def build_session() -> AgentSession:
    """Build the AgentSession for the configured ENGINE_PROFILE."""
    if config.ENGINE_PROFILE == "cascade":
        return _build_cascade()
    if config.ENGINE_PROFILE == "realtime":
        return _build_realtime()
    raise ValueError(
        f"Unknown ENGINE_PROFILE={config.ENGINE_PROFILE!r}. Use 'cascade' or 'realtime'."
    )
