"""Central config for the check-in caller — the only place reading env vars.

Flat on purpose: this whole app is a handful of scripts in one folder, so
there is no package to import engines/agents from. Everything telephony +
model related is read here once and imported everywhere else.

Keys are intentionally the SAME names Rock Scheduler uses so a single .env
(or the shared Twilio number + LiveKit SIP trunk) drops straight in.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Engine selection: cascade | realtime (same knob as Rock) ---
ENGINE_PROFILE: str = os.getenv("ENGINE_PROFILE", "cascade")

# --- Cascade: Deepgram STT -> OpenAI LLM -> Cartesia TTS ---
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
CARTESIA_VOICE: str = os.getenv("CARTESIA_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")

# --- Realtime (OpenAI speech-to-speech), if ENGINE_PROFILE=realtime ---
REALTIME_MODEL: str = os.getenv("REALTIME_MODEL", "gpt-realtime-mini")
REALTIME_VOICE: str = os.getenv("REALTIME_VOICE", "marin")

# --- Provider keys. Empty Deepgram/Cartesia => route via LiveKit Inference ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")

# --- LiveKit transport + SIP (reuse Rock's arya-hack project / trunk) ---
LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")

# This worker's dispatch name. Keep it DISTINCT from Rock's AGENT_NAME so
# outbound check-in calls land here and not on the scheduler worker.
AGENT_NAME: str = os.getenv("AGENT_NAME", "checkin-agent")

# --- Twilio (shared voice number) + LiveKit outbound SIP trunk (shared) ---
TWILIO_PHONE_NUMBER: str = (
    os.getenv("TWILIO_PHONE_NUMBER", "") or os.getenv("TWILIO_VOICE_NUMBER", "")
)
SIP_OUTBOUND_TRUNK_ID: str = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK_ID", "")

# --- Turn detection / endpointing tuning (cascade only) ---
# How long to wait after speech before committing a turn. The turn detector
# picks a delay between these based on end-of-turn confidence:
#   confident  -> ~min_delay      unsure -> ~max_delay
# Library defaults are 0.5 / 3.0; we cut max_delay so an "unsure" pause
# doesn't stall the agent for 3 seconds.
EOT_MIN_DELAY: float = float(os.getenv("EOT_MIN_DELAY", "0.5"))
EOT_MAX_DELAY: float = float(os.getenv("EOT_MAX_DELAY", "1.5"))
# End-of-turn probability below this => "user probably not done" => wait
# max_delay. 0 keeps the model's per-language default (en≈0.3857). Lower it
# to make the agent commit turns sooner (snappier, but riskier mid-sentence).
EOT_UNLIKELY_THRESHOLD: float = float(os.getenv("EOT_UNLIKELY_THRESHOLD", "0"))
# If the agent briefly stops for a false interruption, how long before it
# resumes. Default 2.0s caused ~6s stalls; 1.0s recovers faster.
FALSE_INTERRUPTION_TIMEOUT: float = float(os.getenv("FALSE_INTERRUPTION_TIMEOUT", "1.0"))

# --- Vet front desk escalation: warm transfer to the on-call vet ---
VET_ESCALATION_NUMBER: str = os.getenv("VET_ESCALATION_NUMBER", "+19296799935")
# Hold music played to the caller while the vet is being connected.
# Relative paths are resolved against this folder. Empty -> LiveKit default clip.
HOLD_MUSIC_FILE: str = os.getenv(
    "HOLD_MUSIC_FILE",
    "hold_music.mp3",  # Baha Men, trimmed to start at 0:17
)


def hold_music_path() -> str | None:
    """Absolute path to the hold-music file, or None to use the default clip."""
    if not HOLD_MUSIC_FILE:
        return None
    p = HOLD_MUSIC_FILE
    if not os.path.isabs(p):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
    return p if os.path.isfile(p) else None

# --- Observability ---
DEBUG_TRANSCRIPTS: bool = _env_bool("DEBUG_TRANSCRIPTS", False)


def startup_summary() -> str:
    """One honest line per subsystem so you see what will actually run."""
    stt = "deepgram plugin" if DEEPGRAM_API_KEY else "livekit inference (deepgram/nova-3)"
    tts = "cartesia plugin" if CARTESIA_API_KEY else "livekit inference (cartesia/sonic-3)"
    return "\n".join([
        f"  engine    : {ENGINE_PROFILE}",
        f"  livekit   : {LIVEKIT_URL or 'MISSING'}",
        f"  cascade   : stt={stt} | llm={LLM_MODEL} | tts={tts}",
        (
            f"  turn      : endpoint={EOT_MIN_DELAY}-{EOT_MAX_DELAY}s | "
            f"unlikely={EOT_UNLIKELY_THRESHOLD or 'default'} | "
            f"false-int-timeout={FALSE_INTERRUPTION_TIMEOUT}s"
        ),
        f"  realtime  : {REALTIME_MODEL} voice={REALTIME_VOICE}",
        (
            f"  telephony : agent={AGENT_NAME or '(auto-dispatch)'} | "
            f"number={TWILIO_PHONE_NUMBER or 'MISSING'} | "
            f"out-trunk={SIP_OUTBOUND_TRUNK_ID or 'MISSING'}"
        ),
    ])
