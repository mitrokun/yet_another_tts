import logging
import asyncio
from typing import Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .speech_tts import SpeechTTS
from .sentence_boundary import SentenceBoundaryDetector

log = logging.getLogger(__name__)

class SpeechEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args,
        speech_tts: SpeechTTS,
        voice_map: dict, # Map теперь просто name -> name для проверки
        default_speaker_name: str,
        default_speech_rate: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.speech_tts = speech_tts
        self.voice_map = voice_map
        self.default_speaker_name = default_speaker_name
        self.default_speech_rate = default_speech_rate

        self.is_streaming: bool = False
        self.sbd: Optional[SentenceBoundaryDetector] = None
        self._synthesize: Optional[Synthesize] = None
        
        log.debug(f"Handler initialized. Default Speaker: {self.default_speaker_name}")

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            log.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    return True
                synthesize = Synthesize.from_event(event)
                return await self._handle_synthesize(synthesize)

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                log.debug(f"Text stream started: voice={stream_start.voice}")
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                assert self.sbd is not None
                stream_chunk = SynthesizeChunk.from_event(event)

                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    log.debug(f"Synthesizing stream sentence: {sentence}")
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)
                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                assert self.sbd is not None
                
                final_text = self.sbd.finish()
                if final_text:
                    self._synthesize.text = final_text
                    await self._handle_synthesize(self._synthesize)

                await self.write_event(SynthesizeStopped().event())
                self.is_streaming = False
                self.sbd = None
                self._synthesize = None
                log.debug("Text stream stopped")
                return True

            log.warning("Unexpected event type: %s", event.type)
            return True

        except Exception as e:
            log.error(f"Error handling event: {e}", exc_info=True)
            await self.write_event(Error(text=str(e), code=e.__class__.__name__).event())
            self.is_streaming = False
            self.sbd = None
            self._synthesize = None
            return False


    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        if not synthesize.text:
            return True

        requested_voice_name = synthesize.voice.name if synthesize.voice else None
        speaker_name = self.default_speaker_name
        speech_rate = self.default_speech_rate

        # Проверяем, есть ли запрошенный голос в нашем списке
        if requested_voice_name:
            if requested_voice_name in self.voice_map:
                speaker_name = requested_voice_name
            else:
                log.warning(f"Voice '{requested_voice_name}' not found. Using default '{self.default_speaker_name}'.")

        if hasattr(synthesize, 'speech_rate') and synthesize.speech_rate is not None:
            speech_rate = synthesize.speech_rate
        
        text = " ".join(synthesize.text.strip().splitlines())
        
        # Вызов синтеза Silero
        audio_bytes = await self.speech_tts.synthesize(
            text=text, speaker_name=speaker_name, speech_rate=speech_rate
        )

        if audio_bytes is None:
            log.error(f"Synthesis failed for text: {text[:50]}...")
            await self.write_event(Error(text="TTS synthesis failed").event())
            return True

        try:
            rate = self.speech_tts.sample_rate
            width = self.speech_tts.sample_width
            channels = self.speech_tts.channels

            await self.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )

            bytes_per_sample = width * channels
            bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
            
            if bytes_per_chunk > 0:
                for i in range(0, len(audio_bytes), bytes_per_chunk):
                    chunk = audio_bytes[i : i + bytes_per_chunk]
                    await self.write_event(
                        AudioChunk(audio=chunk, rate=rate, width=width, channels=channels).event()
                    )
            elif len(audio_bytes) > 0:
                await self.write_event(
                    AudioChunk(audio=audio_bytes, rate=rate, width=width, channels=channels).event()
                )

            await self.write_event(AudioStop().event())
        except Exception as e:
            log.error(f"Error streaming audio: {e}", exc_info=True)

        return True