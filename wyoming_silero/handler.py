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
        voice_map: dict,
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
            return True

        try:
            # 1. ЛЕГАСИ СИНТЕЗ (Целый текст)
            if Synthesize.is_type(event.type):
                # Если стриминг активирован, игнорируем легаси-событие, 
                # чтобы не было двойного звука.
                if self.is_streaming:
                    log.debug("Streaming is active, skipping legacy Synthesize event.")
                    return True
                
                synthesize = Synthesize.from_event(event)
                return await self._handle_synthesize(synthesize)

            # 2. НАЧАЛО СТРИМИНГА (По предложениям)
            if SynthesizeStart.is_type(event.type):
                # Если стриминг выключен ключом в консоли, НЕ ставим флаг is_streaming
                if not self.cli_args.streaming:
                    log.debug("Streaming is disabled by CLI, ignoring SynthesizeStart.")
                    return True

                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                stream_start = SynthesizeStart.from_event(event)
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                log.debug(f"Text stream started (voice: {stream_start.voice})")
                return True

            # 3. ПОЛУЧЕНИЕ КУСОЧКА ТЕКСТА
            if SynthesizeChunk.is_type(event.type):
                if not self.is_streaming:
                    return True
                
                assert self._synthesize is not None
                assert self.sbd is not None
                
                stream_chunk = SynthesizeChunk.from_event(event)
                # Детектор границ предложений выдает готовые фразы по мере накопления текста
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    log.debug(f"Synthesizing stream sentence: {sentence}")
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)
                return True

            # 4. КОНЕЦ СТРИМИНГА
            if SynthesizeStop.is_type(event.type):
                if not self.is_streaming:
                    return True
                
                assert self._synthesize is not None
                assert self.sbd is not None
                
                # Дочитываем то, что осталось в буфере
                final_text = self.sbd.finish()
                if final_text:
                    self._synthesize.text = final_text
                    await self._handle_synthesize(self._synthesize)

                await self.write_event(SynthesizeStopped().event())
                self.is_streaming = False # Сбрасываем флаг после завершения
                self.sbd = None
                self._synthesize = None
                log.debug("Text stream stopped")
                return True

            return True

        except Exception as e:
            log.error(f"Error handling event: {e}", exc_info=True)
            await self.write_event(Error(text=str(e), code=e.__class__.__name__).event())
            self.is_streaming = False
            return False

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        """Общий метод для отправки текста в движок Silero и передачи аудио клиенту"""
        if not synthesize.text:
            return True

        requested_voice_name = synthesize.voice.name if synthesize.voice else None
        speaker_name = self.default_speaker_name
        speech_rate = self.default_speech_rate

        if requested_voice_name and requested_voice_name in self.voice_map:
            speaker_name = requested_voice_name

        text = " ".join(synthesize.text.strip().splitlines())
        
        # Вызов вашего класса SpeechTTS
        audio_bytes = await self.speech_tts.synthesize(
            text=text, speaker_name=speaker_name, speech_rate=speech_rate
        )

        if not audio_bytes:
            return True

        # Отправка аудио в Wyoming
        rate = self.speech_tts.sample_rate
        width = self.speech_tts.sample_width
        channels = self.speech_tts.channels

        await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        
        bytes_per_chunk = width * channels * self.cli_args.samples_per_chunk
        for i in range(0, len(audio_bytes), bytes_per_chunk):
            await self.write_event(
                AudioChunk(audio=audio_bytes[i : i + bytes_per_chunk], 
                           rate=rate, width=width, channels=channels).event()
            )

        await self.write_event(AudioStop().event())
        return True