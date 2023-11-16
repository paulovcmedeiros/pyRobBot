"""Code related to the voice chat feature."""
import contextlib
import io
import queue
from collections import deque
from datetime import datetime

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

from pyrobbot.chat_configs import VoiceChatConfigs

from .chat import Chat
from .openai_utils import CannotConnectToApiError, retry_api_call

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


class VoiceChat(Chat):
    """Class for converting text to speech and speech to text."""

    def __init__(self, configs: VoiceChatConfigs = None):
        """Initializes a chat instance."""
        if not _sounddevice_imported:
            raise ImportError(
                "Module `sounddevice`, needed for audio recording, is not available."
            )

        if configs is None:
            configs = VoiceChatConfigs()
        super().__init__(configs=configs)

        self.mixer = pygame.mixer
        self.vad = webrtcvad.Vad(2)
        self.mixer.init()

    def start(self):
        """Start the chat."""
        # ruff: noqa: T201
        self.speak(self._translate(self.initial_greeting))
        try:
            previous_question_answered = True
            while True:
                if previous_question_answered:
                    logger.info(f"{self.assistant_name}> Listening...")
                question = self.listen()
                if not question:
                    previous_question_answered = False
                    continue
                logger.info(f"{self.assistant_name}> Let me think...")
                answer = "".join(self.respond_user_prompt(prompt=question))
                logger.info(f"{self.assistant_name}> Ok, here we go:")
                self.speak(answer)
                previous_question_answered = True
        except (KeyboardInterrupt, EOFError):
            print("", end="\r")
            logger.info("Leaving chat.")
        except CannotConnectToApiError as error:
            print(f"{self.api_connection_error_msg}\n")
            logger.error("Leaving chat: {}", error)

    def speak(self, text):
        """Convert text to speech."""
        logger.debug("Converting text to speech...")
        if self.tts_engine == "openai" and _pydub_imported:
            tts_wav__buffer = self._tts_openai(text)
        else:
            tts_wav__buffer = self._tts_google(text)
        logger.debug("Done converting text to speech.")

        # Play the audio file
        speech_sound = self._wav_buffer_to_sound(wav_buffer=tts_wav__buffer)
        _channel = speech_sound.play()
        while self._assistant_still_talking():
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
        text = self._wav_buffer_to_text(wav_buffer=raw_buffer)
        logger.debug("Done converting audio to text.")

        return text

    def _assistant_still_talking(self):
        """Check if the assistant is still talking."""
        return self.mixer.get_busy()

    @retry_api_call()
    def _tts_openai(self, text):
        """Convert text to speech using OpenAI's TTS."""
        text = text.strip()
        client = OpenAI()

        openai_tts_model = "tts-1"

        for db in [
            self.general_token_usage_db,
            self.token_usage_db,
        ]:
            db.insert_data(model=openai_tts_model, n_input_tokens=len(text))

        response = client.audio.speech.create(
            input=text,
            model=openai_tts_model,
            voice=self.openai_tts_voice,
            response_format="mp3",
        )

        mp3_buffer = io.BytesIO()
        for mp3_stream_chunk in response.iter_bytes(chunk_size=4096):
            mp3_buffer.write(mp3_stream_chunk)
        mp3_buffer.seek(0)

        wav_buffer = io.BytesIO()
        sound = pydub.AudioSegment.from_mp3(mp3_buffer)
        # Increase the default volume, the default is a bit to quiet
        volume_increase_db = 6
        sound += volume_increase_db
        sound.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        return wav_buffer

    def _tts_google(self, text):
        """Convert text to speech using Google's TTS."""
        tts = gTTS(text.strip(), lang=self.language)
        wav_buffer = io.BytesIO()
        tts.write_to_fp(wav_buffer)
        wav_buffer.seek(0)
        return wav_buffer

    def _wav_buffer_to_sound(self, wav_buffer):
        """Create a pygame sound object from a BytesIO object."""
        return self.mixer.Sound(wav_buffer)

    def _wav_buffer_to_text(self, wav_buffer):
        """Use SpeechRecognition to convert the audio to text."""
        wav_buffer.seek(0)  # Reset the file pointer to the beginning of the file
        r = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
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
