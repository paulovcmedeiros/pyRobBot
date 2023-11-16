"""Code related to the voice chat feature."""
import contextlib
import io
import queue
import re
import threading
from collections import deque
from datetime import datetime

import chime
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

    default_configs = VoiceChatConfigs()

    def __init__(self, configs: VoiceChatConfigs = default_configs):
        """Initializes a chat instance."""
        if not _sounddevice_imported:
            raise ImportError(
                "Module `sounddevice`, needed for audio recording, is not available."
            )

        super().__init__(configs=configs)

        self.mixer = pygame.mixer
        self.vad = webrtcvad.Vad(2)
        self.mixer.init()
        chime.theme("big-sur")

        # Create queues for TTS processing and speech playing
        self.tts_conversion_queue = queue.Queue()
        self.play_speech_queue = queue.Queue()
        # Create threads to watch the TTS and speech playing queues
        self.tts_conversion_watcher_thread = threading.Thread(
            target=self.get_tts, args=(self.tts_conversion_queue,), daemon=True
        )
        self.play_speech_thread = threading.Thread(
            target=self.speak, args=(self.play_speech_queue,), daemon=True
        )
        self.tts_conversion_watcher_thread.start()
        self.play_speech_thread.start()

    def start(self):
        """Start the chat."""
        # ruff: noqa: T201
        if not self.skip_initial_greeting:
            self.tts_conversion_queue.put(self.initial_greeting)

        try:
            previous_question_answered = True
            while True:
                # Wait for all items in the queue to be processed
                self.tts_conversion_queue.join()
                self.play_speech_queue.join()
                if previous_question_answered:
                    chime.warning()
                    previous_question_answered = False
                    logger.debug(f"{self.assistant_name}> Listening...")

                question = self.listen().strip()
                if not question:
                    continue

                # Check for the exit expressions
                if any(
                    _get_lower_alphanumeric(question).startswith(
                        _get_lower_alphanumeric(expr)
                    )
                    for expr in self.exit_expressions
                ):
                    chime.theme("material")
                    chime.error()
                    logger.debug(f"{self.assistant_name}> Goodbye!")
                    break

                chime.success()
                logger.debug(f"{self.assistant_name}> Getting response...")
                sentence = ""
                for answer_chunk in self.respond_user_prompt(prompt=question):
                    sentence += answer_chunk
                    if answer_chunk.strip().endswith(("?", "!", ".")):
                        # Send sentence for TTS even if the request hasn't yet finished
                        self.tts_conversion_queue.put(sentence)
                        sentence = ""
                if sentence:
                    self.tts_conversion_queue.put(sentence)

                previous_question_answered = True

        except (KeyboardInterrupt, EOFError):
            chime.info()
            print("", end="\r")
            logger.debug("Leaving chat.")
        except CannotConnectToApiError as error:
            chime.error()
            print(f"{self.api_connection_error_msg}\n")
            logger.error("Leaving chat: {}", error)

    def get_tts(self, text_queue: queue.Queue):
        """Convert text to a pygame Sound object."""
        while True:
            text = text_queue.get()
            logger.debug("Received for TTS: '{}'", text)

            if self.tts_engine == "openai" and _pydub_imported:
                tts_wav_buffer = self._tts_openai(text)
            else:
                tts_wav_buffer = self._tts_google(text)

            # Convert wav buffer to sound
            speech_sound = self._wav_buffer_to_sound(wav_buffer=tts_wav_buffer)
            self.play_speech_queue.put(speech_sound)

            logger.debug("Done with TTS for '{}'", text)
            text_queue.task_done()

    def speak(self, sound_obj_queue: queue.Queue):
        """Reproduce audio from a pygame Sound object."""
        while True:
            sound = sound_obj_queue.get()
            _channel = sound.play()
            while self._assistant_still_talking():
                pygame.time.wait(100)
            sound_obj_queue.task_done()

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
                    wav_buffer = _np_array_to_wav_in_memory(
                        new_data, sample_rate=self.sample_rate
                    )
                    vad_thinks_this_chunk_is_speech = self.vad.is_speech(
                        wav_buffer, self.sample_rate
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
        volume_increase_db = 8
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


def _get_lower_alphanumeric(string: str):
    """Return a string with only lowercase alphanumeric characters."""
    return re.sub("[^0-9a-zA-Z]+", " ", string.strip().lower())
