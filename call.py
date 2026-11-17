"""Place one outbound check-in call: dispatch the agent, then dial via SIP.

Agent-first ordering so the callee never answers into an empty room.

Usage:
  python call.py +1XXXXXXXXXX "Alex"
  python call.py +1XXXXXXXXXX "Alex" "Did you sleep well?" "Any pain today?"

The 1st arg is the E.164 number, the 2nd is the callee name, and any extra
args become the checklist (otherwise checkin_agent.DEFAULT_QUESTIONS is used).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid

from livekit import api

import config

log = logging.getLogger("checkin.call")


async def place_call(to: str, name: str = "there",
                     questions: list[str] | None = None) -> dict:
    """Ring `to` (E.164) and connect the check-in agent. Never raises."""
    if not config.SIP_OUTBOUND_TRUNK_ID:
        return {"ok": False, "to": to, "error": "LIVEKIT_SIP_OUTBOUND_TRUNK_ID not set"}

    metadata = json.dumps({"name": name, "questions": questions})
    room = f"checkin-{uuid.uuid4().hex[:8]}"

    lk = api.LiveKitAPI()
    try:
        if config.AGENT_NAME:
            await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
                agent_name=config.AGENT_NAME, room=room, metadata=metadata,
            ))
        participant = await lk.sip.create_sip_participant(api.CreateSIPParticipantRequest(
            sip_trunk_id=config.SIP_OUTBOUND_TRUNK_ID,
            sip_call_to=to,
            sip_number=config.TWILIO_PHONE_NUMBER,
            room_name=room,
            participant_identity=f"phone-{to}",
            wait_until_answered=False,
        ))
        log.info("outbound check-in -> %s room=%s", to, room)
        return {"ok": True, "to": to, "room": room,
                "participant": participant.participant_identity}
    except Exception as exc:  # noqa: BLE001 — surface as a result, not a crash
        log.error("outbound check-in -> %s FAILED: %s", to, exc)
        return {"ok": False, "to": to, "room": room, "error": str(exc)}
    finally:
        await lk.aclose()


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    to = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "there"
    questions = list(sys.argv[3:]) or None
    result = asyncio.run(place_call(to, name=name, questions=questions))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
