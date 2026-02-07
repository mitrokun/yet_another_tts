import os
from argparse import ArgumentParser
import asyncio
import contextlib
import logging
import sys
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .handler import SpeechEventHandler
from .speech_tts import SpeechTTS

log = logging.getLogger(__name__)

# --- Константы ---
DEFAULT_SPEAKER_NAME = "xenia"
DEFAULT_SPEECH_RATE = 1.0
MODEL_LANGUAGE = "ru-RU"
DEFAULT_VOICE_VERSION = "5.1"
ATTRIBUTION_NAME = "Silero"
ATTRIBUTION_URL = "https://github.com/snakers4/silero-models"
PROGRAM_NAME = "silero-tts-wyoming"
PROGRAM_DESCRIPTION = "Wyoming server for Silero TTS"
PROGRAM_VERSION = "1.0"

SILERO_SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]

async def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--uri", default="tcp://0.0.0.0:10208", help="unix:// or tcp://"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed output.",
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)

    parser.add_argument(
        "--default-speaker",
        type=str,
        default=DEFAULT_SPEAKER_NAME,
        help=f"Default speaker name. Options: {SILERO_SPEAKERS}"
    )

    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable audio streaming on sentence boundaries.",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    log.info("Starting Silero TTS Wyoming Server")

    try:
        speech_tts_instance = SpeechTTS()
    except Exception as e:
        log.critical(f"Failed to initialize TTS engine: {e}")
        sys.exit(1)

    if args.default_speaker not in SILERO_SPEAKERS:
        log.warning(f"Default speaker '{args.default_speaker}' not found in {SILERO_SPEAKERS}. Using '{DEFAULT_SPEAKER_NAME}'.")
        args.default_speaker = DEFAULT_SPEAKER_NAME

    voices = []
    voice_map = {name: name for name in SILERO_SPEAKERS}

    for name in SILERO_SPEAKERS:
        voices.append(TtsVoice(
            name=name,
            description=name.capitalize(),
            attribution=Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL),
            installed=True,
            version=DEFAULT_VOICE_VERSION,
            languages=[MODEL_LANGUAGE]
        ))

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name=PROGRAM_NAME,
                description=PROGRAM_DESCRIPTION,
                attribution=Attribution(
                    name=ATTRIBUTION_NAME,
                    url=ATTRIBUTION_URL
                ),
                installed=True,
                version=PROGRAM_VERSION,
                voices=voices,
                supports_synthesize_streaming=args.streaming,
            )
        ]
    )

    handler_factory = partial(
        SpeechEventHandler,
        wyoming_info,
        args,
        speech_tts_instance,
        voice_map,
        args.default_speaker,
        DEFAULT_SPEECH_RATE,
    )

    server = AsyncServer.from_uri(args.uri)
    log.info(f"Server ready and listening at {args.uri}")
    await server.run(handler_factory)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(main())
    finally:
        log.info("Server shutting down.")