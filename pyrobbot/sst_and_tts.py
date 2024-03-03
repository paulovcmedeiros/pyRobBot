"""Code related to speech-to-text and text-to-speech conversions."""

import io
import socket
import uuid
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import speech_recognition as sr
from gtts import gTTS
from loguru import logger
from openai import OpenAI
from pydub import AudioSegment

from .general_utils import retry
from .tokens import TokenUsageDatabase


@dataclass
class SpeechAndTextConfigs:
    """Configs for speech-to-text and text-to-speech."""

    openai_client: OpenAI
    general_token_usage_db: TokenUsageDatabase
    token_usage_db: TokenUsageDatabase
    engine: Literal["openai", "google"] = "google"
    language: str = "en"
    timeout: int = 10


@dataclass
class SpeechToText(SpeechAndTextConfigs):
    """Class for converting speech to text."""

    speech: AudioSegment = None
    _text: str = field(init=False, default="")

    def __post_init__(self):
        if not self.speech:
            self.speech = AudioSegment.silent(duration=0)
        self.recogniser = sr.Recognizer()
        self.recogniser.operation_timeout = self.timeout

        wav_buffer = io.BytesIO()
        self.speech.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        with sr.AudioFile(wav_buffer) as source:
            self.audio_data = self.recogniser.listen(source)

    @property
    def text(self) -> str:
        """Return the text from the speech."""
        if not self._text:
            self._text = self._stt()
        return self._text

    def _stt(self) -> str:
        """Perform speech-to-text."""
        if not self.speech:
            logger.debug("No speech detected")
            return ""

        if self.engine == "openai":
            stt_function = self._stt_openai
            fallback_stt_function = self._stt_google
            fallback_name = "google"
        else:
            stt_function = self._stt_google
            fallback_stt_function = self._stt_openai
            fallback_name = "openai"

        conversion_id = uuid.uuid4()
        logger.debug(
            "Converting audio to text ({} STT). Process {}.", self.engine, conversion_id
        )
        try:
            rtn = stt_function()
        except (
            ConnectionResetError,
            socket.timeout,
            sr.exceptions.RequestError,
        ) as error:
            logger.error(error)
            logger.error(
                "{}: Can't communicate with `{}` speech-to-text API right now",
                conversion_id,
                self.engine,
            )
            logger.warning(
                "{}: Trying to use `{}` STT instead", conversion_id, fallback_name
            )
            rtn = fallback_stt_function()
        except sr.exceptions.UnknownValueError:
            logger.opt(colors=True).debug(
                "<yellow>{}: Can't understand audio</yellow>", conversion_id
            )
            rtn = ""

        self._text = rtn.strip()
        logger.opt(colors=True).debug(
            "<yellow>{}: Done with STT: {}</yellow>", conversion_id, self._text
        )

        return self._text

    @retry()
    def _stt_openai(self):
        """Perform speech-to-text using OpenAI's API."""
        wav_buffer = io.BytesIO(self.audio_data.get_wav_data())
        wav_buffer.name = "audio.wav"
        with wav_buffer as audio_file_buffer:
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_buffer,
                language=self.language.split("-")[0],  # put in ISO-639-1 format
                prompt=f"The language is {self.language}. "
                "Do not transcribe if you think the audio is noise.",
            )

        for db in [
            self.general_token_usage_db,
            self.token_usage_db,
        ]:
            db.insert_data(
                model="whisper-1",
                n_input_tokens=int(np.ceil(self.speech.duration_seconds)),
            )

        return transcript.text

    def _stt_google(self):
        """Perform speech-to-text using Google's API."""
        return self.recogniser.recognize_google(
            audio_data=self.audio_data, language=self.language
        )


@dataclass
class TextToSpeech(SpeechAndTextConfigs):
    """Class for converting text to speech."""

    text: str = ""
    openai_tts_voice: str = ""
    _speech: AudioSegment = field(init=False, default=None)

    def __post_init__(self):
        self.text = self.text.strip()

    @property
    def speech(self) -> AudioSegment:
        """Return the speech from the text."""
        if not self._speech:
            self._speech = self._tts()
        return self._speech

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate of the speech."""
        self._speech = self.speech.set_frame_rate(sample_rate)

    def _tts(self):
        logger.debug("Running {} TTS on text '{}'", self.engine, self.text)
        rtn = self._tts_openai() if self.engine == "openai" else self._tts_google()
        logger.debug("Done with TTS for '{}'", self.text)

        return rtn

    def _tts_openai(self) -> AudioSegment:
        """Convert text to speech using OpenAI's TTS. Return an AudioSegment object."""
        openai_tts_model = "tts-1"

        @retry()
        def _create_speech(*args, **kwargs):
            for db in [
                self.general_token_usage_db,
                self.token_usage_db,
            ]:
                db.insert_data(model=openai_tts_model, n_input_tokens=len(self.text))
            return self.openai_client.audio.speech.create(*args, **kwargs)

        response = _create_speech(
            input=self.text,
            model=openai_tts_model,
            voice=self.openai_tts_voice,
            response_format="mp3",
            timeout=self.timeout,
        )

        mp3_buffer = io.BytesIO()
        for mp3_stream_chunk in response.iter_bytes(chunk_size=4096):
            mp3_buffer.write(mp3_stream_chunk)
        mp3_buffer.seek(0)

        audio = AudioSegment.from_mp3(mp3_buffer)
        audio += 8  # Increase volume a bit
        return audio

    def _tts_google(self) -> AudioSegment:
        """Convert text to speech using Google's TTS. Return a WAV BytesIO object."""
        tts = gTTS(self.text, lang=self.language)
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        return AudioSegment.from_mp3(mp3_buffer)
