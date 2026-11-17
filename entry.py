"""LiveKit worker entrypoint for the check-in caller.

Run it one of two ways:
  python entry.py console   # talk to it from your terminal mic (no phone)
  python entry.py dev       # register the worker; outbound calls dispatch here

Dispatch metadata shapes the call. channels/call.py sends:
  {"name": "Alex"}
The checklist itself is fixed in checklist.py (3 fields for now).
"""

import json
import logging

from livekit.agents import (
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    UserInputTranscribedEvent,
    cli,
)
from livekit.agents.llm import ChatMessage

import config
from checkin_agent import build_checkin_agent
from engine import build_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-16s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("checkin.entry")

server = AgentServer()


def wire_logging(session: AgentSession) -> None:
    """Log committed turns + per-turn latency; raw transcripts are gated."""
    if config.DEBUG_TRANSCRIPTS:
        @session.on("user_input_transcribed")
        def _on_transcript(ev: UserInputTranscribedEvent) -> None:
            marker = "FINAL  " if ev.is_final else "interim"
            log.info("transcript [%s] %s", marker, ev.transcript)

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if not isinstance(ev.item, ChatMessage):
            return
        log.info("committed  [%s] %s", ev.item.role, ev.item.text_content)
        _log_turn_latency(ev.item)


def _log_turn_latency(item: ChatMessage) -> None:
    metrics = item.metrics or {}
    wanted = (
        ("transcription_delay", "stt_delay"),
        ("end_of_turn_delay", "eou_delay"),
        ("llm_node_ttft", "llm_ttft"),
        ("tts_node_ttfb", "tts_ttfb"),
        ("e2e_latency", "e2e"),
    )
    parts = [
        f"{label}={metrics[key]:.3f}s"
        for key, label in wanted
        if metrics.get(key) is not None
    ]
    if parts:
        log.info("latency    [%s] %s", item.role, " ".join(parts))


@server.rtc_session(agent_name=config.AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    name = meta.get("name") or "there"

    session = build_session()
    wire_logging(session)

    agent = build_checkin_agent(name=name)
    log.info("starting CheckinAgent for %r on engine %r", name, config.ENGINE_PROFILE)

    # record=False keeps audio/transcripts off LiveKit Cloud by default.
    await session.start(agent=agent, room=ctx.room, record=False)
    await session.generate_reply(
        instructions=(
            f"Greet {name} warmly as Rock, say you're calling for a quick "
            "check-in, and ask the first checklist question."
        )
    )


if __name__ == "__main__":
    print("ROCK Check-in caller\n" + config.startup_summary())
    cli.run_app(server)
