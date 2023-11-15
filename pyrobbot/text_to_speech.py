"""Functions for converting text to speech and speech to text."""
import contextlib
import io
import queue
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pydub
import pygame
import scipy.io.wavfile as wav
import soundfile as sf
import speech_recognition as sr
import webrtcvad
from gtts import gTTS
from loguru import logger
from openai import OpenAI

from .openai_utils import retry_api_call

if TYPE_CHECKING:
    from .chat import Chat

try:
    import sounddevice as sd

    _sounddevice_imported = True
except OSError as error:
    logger.error(error)
    logger.error(
        "Can't use module `sounddevice`. Please check your system's PortAudio install."
    )
    _sounddevice_imported = False

try:
    # Test if AudioSegment.from_mp3() can be used
    with contextlib.suppress(pydub.exceptions.CouldntDecodeError):
        pydub.AudioSegment.from_mp3(io.BytesIO())
except (ImportError, OSError, FileNotFoundError) as error:
    logger.error(
        "{}. Can't use module `pydub`. Please check your system's ffmpeg install.", error
    )
    logger.warning("Using Google's TTS instead of OpenAI's, which requires `pydub`.")
    _pydub_imported = False
else:
    _pydub_imported = True


@dataclass
class LiveAssistant:
    """Class for converting text to speech and speech to text."""

    parent_chat: "Chat"
    # How much time user should be inactive for the assistant to stop listening
    inactivity_timeout_seconds: int = 2
    # Accept audio as speech if the likelihood is above this threshold
    speech_likelihood_threshold: float = 0.85
    # Params for audio capture
    sample_rate: int = 32000  # Hz
    frame_duration: int = 30  # milliseconds

    def __post_init__(self):
        if not _sounddevice_imported:
            raise ImportError(
                "Module `sounddevice`, needed for audio recording, is not available."
            )

        webrtcvad_restrictions = {
            "sample_rate": [8000, 16000, 32000, 48000],
            "frame_duration": [10, 20, 30],
        }
        for attr, allowed_values in webrtcvad_restrictions.items():
            passed_value = getattr(self, attr)
            if passed_value not in allowed_values:
                raise ValueError(
                    f"{attr} must be one of: {allowed_values}. Got '{passed_value}'."
                )

        self.language = self.parent_chat.language_speech
        self.mixer = pygame.mixer
        self.vad = webrtcvad.Vad(2)

        self.mixer.init()

    def get_sound_from_wav_buffer(self, wav_buffer):
        """Create a pygame sound object from a BytesIO object."""
        return self.mixer.Sound(wav_buffer)

    def still_talking(self):
        """Check if the assistant is still talking."""
        return self.mixer.get_busy()

    @retry_api_call()
    def tts_openai(self, text):
        """Convert text to speech using OpenAI's TTS."""
        text = text.strip()
        client = OpenAI()

        openai_tts_model = "tts-1"

        for db in [
            self.parent_chat.general_token_usage_db,
            self.parent_chat.token_usage_db,
        ]:
            db.insert_data(model=openai_tts_model, n_input_tokens=len(text))

        response = client.audio.speech.create(
            input=text,
            model=openai_tts_model,
            voice=self.parent_chat.openai_tts_voice,
            response_format="mp3",
        )

        mp3_buffer = io.BytesIO()
        for mp3_stream_chunk in response.iter_bytes(chunk_size=4096):
            mp3_buffer.write(mp3_stream_chunk)
        mp3_buffer.seek(0)

        wav_buffer = io.BytesIO()
        sound = pydub.AudioSegment.from_mp3(mp3_buffer)
        sound.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        return wav_buffer

    def tts_google(self, text):
        """Convert text to speech using Google's TTS."""
        tts = gTTS(text.strip(), lang=self.language)
        wav_buffer = io.BytesIO()
        tts.write_to_fp(wav_buffer)
        wav_buffer.seek(0)
        return wav_buffer

    def speak(self, text):
        """Convert text to speech."""
        logger.debug("Converting text to speech...")
        if self.parent_chat.tts_engine == "openai" and _pydub_imported:
            tts_wav__buffer = self.tts_openai(text)
        else:
            tts_wav__buffer = self.tts_google(text)
        logger.debug("Done converting text to speech.")

        # Play the audio file
        speech_sound = self.get_sound_from_wav_buffer(wav_buffer=tts_wav__buffer)
        _channel = speech_sound.play()
        while self.still_talking():
            pygame.time.wait(100)

    def listen(self):
        """Record audio from the microphone, until user stops, and convert it to text."""
        # Adapted from
        # <https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html#
        #  recording-with-arbitrary-duration>
        logger.debug("The assistant is listening...")
        q = queue.Queue()

        def callback(indata, frames, time, status):  # noqa: ARG001
            """This is called (from a separate thread) for each audio block."""
            q.put(indata.copy())

        stream_block_size = int((self.sample_rate * self.frame_duration) / 1000)
        raw_buffer = io.BytesIO()
        with sf.SoundFile(
            raw_buffer,
            mode="x",
            samplerate=self.sample_rate,
            channels=1,
            format="wav",
            subtype="PCM_16",
        ) as audio_file, sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=stream_block_size,
            channels=1,
            callback=callback,
            dtype="int16",
        ):
            # Recording will stop after self.inactivity_timeout_seconds of silence
            voice_activity_detected = deque(
                maxlen=int(
                    (1000.0 * self.inactivity_timeout_seconds) / self.frame_duration
                )
            )
            last_inactivity_checked = datetime.now()
            user_is_speaking = True
            speech_detected = False
            with contextlib.suppress(KeyboardInterrupt):
                while user_is_speaking:
                    new_data = q.get()
                    audio_file.write(new_data)

                    # Gather voice activity samples for the inactivity check
                    vad_thinks_this_chunk_is_speech = self.vad.is_speech(
                        _np_array_to_wav_in_memory(
                            new_data, sample_rate=self.sample_rate
                        ),
                        self.sample_rate,
                    )
                    voice_activity_detected.append(vad_thinks_this_chunk_is_speech)
                    # Decide if user has been inactive for too long
                    now = datetime.now()
                    if (
                        now - last_inactivity_checked
                    ).seconds >= self.inactivity_timeout_seconds:
                        speech_likelihood = 0.0
                        if len(voice_activity_detected) > 0:
                            speech_likelihood = sum(voice_activity_detected) / len(
                                voice_activity_detected
                            )
                        user_is_speaking = (
                            speech_likelihood >= self.speech_likelihood_threshold
                        )
                        if user_is_speaking:
                            speech_detected = True
                        last_inactivity_checked = now

        if not speech_detected:
            logger.debug("No speech detected")
            return ""

        logger.debug("Converting audio to text...")
        text = self._audio_buffer_to_text(byte_io=raw_buffer)
        logger.debug("Done converting audio to text.")

        return text

    def _audio_buffer_to_text(self, byte_io):
        """Use SpeechRecognition to convert the audio to text."""
        byte_io.seek(0)  # Reset the file pointer to the beginning of the file
        r = sr.Recognizer()
        with sr.AudioFile(byte_io) as source:
            audio_data = r.listen(source)

        try:
            return r.recognize_google(audio_data, language=self.language)
        except sr.exceptions.UnknownValueError:
            logger.debug("Could not understand audio")
            return ""


def _np_array_to_wav_in_memory(array: np.ndarray, sample_rate: int):
    """Convert the recorded array to an in-memory wav file."""
    byte_io = io.BytesIO()
    wav.write(byte_io, rate=sample_rate, data=array)
    byte_io.seek(44)  # Skip the WAV header
    return byte_io.read()
