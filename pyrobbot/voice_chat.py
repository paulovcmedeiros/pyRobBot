"""Code related to the voice chat feature."""
import contextlib
import io
import queue
import re
import socket
import threading
from collections import deque
from datetime import datetime
from functools import partial

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

from .chat import Chat
from .chat_configs import VoiceChatConfigs
from .general_utils import retry

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

        self.block_size = int((self.sample_rate * self.frame_duration) / 1000)

        self.mixer = pygame.mixer
        self.mixer.init(frequency=self.sample_rate, channels=1, buffer=self.block_size)

        self.vad = webrtcvad.Vad(2)
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

    def start(self):  # noqa: PLR0912, PLR0915
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
                logger.debug(f"{self.assistant_name}> Heard: '{question}'")

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
                inside_code_block = False
                at_least_one_code_line_written = False
                for answer_chunk in self.respond_user_prompt(prompt=question):
                    fmtd_chunk = answer_chunk.strip(" \n")
                    code_block_start_detected = fmtd_chunk.startswith("``")

                    if code_block_start_detected and not inside_code_block:
                        # Toggle the code block state
                        inside_code_block = True

                    if inside_code_block:
                        code_chunk = answer_chunk
                        if at_least_one_code_line_written:
                            inside_code_block = not fmtd_chunk.endswith("``")  # Code ends
                            if not inside_code_block:
                                code_chunk = answer_chunk.rstrip("`") + "```\n"
                        print(
                            code_chunk,
                            end="" if inside_code_block else "\n",
                            flush=True,
                        )
                        at_least_one_code_line_written = True
                    else:
                        # The answer chunk is to be spoken
                        sentence += answer_chunk
                        if answer_chunk.strip().endswith(("?", "!", ".")):
                            # Send sentence for TTS even if the request hasn't finished
                            self.tts_conversion_queue.put(sentence)
                            sentence = ""

                if sentence:
                    self.tts_conversion_queue.put(sentence)

                if at_least_one_code_line_written:
                    spoken_info_to_user = "The code has been written to the console"
                    spoken_info_to_user = self._translate(spoken_info_to_user)
                    self.tts_conversion_queue.put(spoken_info_to_user)

                previous_question_answered = True

        except (KeyboardInterrupt, EOFError):
            chime.info()
        finally:
            logger.debug("Leaving chat")

    def get_tts(self, text_queue: queue.Queue):
        """Convert text to a pygame Sound object."""
        while True:
            try:
                text = text_queue.get()
                logger.debug("Received for {} TTS: '{}'", self.tts_engine, text)

                if self.tts_engine == "openai" and _pydub_imported:
                    tts_wav_buffer = self._tts_openai(text)
                else:
                    tts_wav_buffer = self._tts_google(text)

                # Dispatch the wav buffer tp be played
                self.play_speech_queue.put(tts_wav_buffer)

                logger.debug("Done with TTS for '{}'", text)
            except Exception as error:  # noqa: PERF203, BLE001
                logger.opt(exception=True).debug(error)
                logger.error(error)
            finally:
                text_queue.task_done()

    def speak(self, wav_buffer_queue: queue.Queue[io.BytesIO]):
        """Reproduce audio from a pygame Sound object."""
        while True:
            try:
                wav_buffer = wav_buffer_queue.get()
                pygame_mixer_sound = self.mixer.Sound(file=wav_buffer)
                _channel = pygame_mixer_sound.play()
                while self._assistant_still_talking():
                    pygame.time.wait(100)
            except Exception as error:  # noqa: PERF203, BLE001
                logger.exception(error)
            finally:
                wav_buffer_queue.task_done()

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
            blocksize=self.block_size,
            channels=1,
            callback=callback,
            dtype="int16",  # int16, i.e., 2 bytes per sample
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

        return self._wav_buffer_to_text(wav_buffer=raw_buffer)

    def _assistant_still_talking(self):
        """Check if the assistant is still talking."""
        return self.mixer.get_busy()

    def _tts_openai(self, text):
        """Convert text to speech using OpenAI's TTS. Return a wav BytesIO object."""
        text = text.strip()
        client = OpenAI(timeout=self.timeout)

        openai_tts_model = "tts-1"

        @retry()
        def _create_speech(*args, **kwargs):
            for db in [
                self.general_token_usage_db,
                self.token_usage_db,
            ]:
                db.insert_data(model=openai_tts_model, n_input_tokens=len(text))
            return client.audio.speech.create(*args, **kwargs)

        response = _create_speech(
            input=text,
            model=openai_tts_model,
            voice=self.openai_tts_voice,
            response_format="mp3",
            timeout=self.timeout,
        )

        mp3_buffer = io.BytesIO()
        for mp3_stream_chunk in response.iter_bytes(chunk_size=4096):
            mp3_buffer.write(mp3_stream_chunk)
        mp3_buffer.seek(0)

        return mp3_ro_wav(mp3_buffer, volume_increase_db=8)

    def _tts_google(self, text):
        """Convert text to speech using Google's TTS. Return a wav BytesIO object."""
        tts = gTTS(text.strip(), lang=self.language)
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        wav_buffer = mp3_ro_wav(mp3_buffer)
        return wav_buffer

    def _wav_buffer_to_text(self, wav_buffer) -> str:
        """Use SpeechRecognition to convert the audio to text."""
        r = sr.Recognizer()
        r.operation_timeout = self.timeout

        get_stt_openai = self._speech_to_text_openai
        get_stt_google = partial(r.recognize_google, language=self.language)
        if self.stt_engine == "openai":
            stt_function = get_stt_openai
            fallback_stt_function = get_stt_google
            fallback_name = "google"
        else:
            stt_function = get_stt_google
            fallback_stt_function = get_stt_openai
            fallback_name = "openai"

        logger.debug("Converting audio to text ({} STT)...", self.stt_engine)
        wav_buffer.seek(0)
        with sr.AudioFile(wav_buffer) as source:
            audio_data = r.listen(source)

        try:
            rtn = stt_function(audio_data)
        except (ConnectionResetError, socket.timeout) as error:
            logger.error(error)
            logger.error(
                "Can't communicate with `{}` speech-to-text API right now",
                self.stt_engine,
            )
            logger.warning("Trying to use `{}` STT instead", fallback_name)
            rtn = fallback_stt_function(audio_data)
        except sr.exceptions.UnknownValueError:
            rtn = ""

        rtn = rtn.strip()
        if rtn:
            logger.debug("Heard: '{}'", rtn)
        else:
            logger.debug("Could not understand audio")

        return rtn

    @retry()
    def _speech_to_text_openai(self, audio_data: sr.AudioData):
        """Convert audio data to text using OpenAI's API."""
        new_buffer = io.BytesIO(audio_data.get_wav_data())
        new_buffer.name = "audio.wav"
        with new_buffer as audio_file_buffer:
            transcript = OpenAI(timeout=self.timeout).audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_buffer,
                language=self.language.split("-")[0],  # put in ISO-639-1 format
                prompt=f"The language is {self.language}. "
                "Do not transcribe if you think it is noise.",
            )

        # Register the number of audio minutes used for the transcription
        sound_length_in_seconds = len(audio_data.get_raw_data()) / (
            audio_data.sample_width * audio_data.sample_rate
        )
        for db in [
            self.general_token_usage_db,
            self.token_usage_db,
        ]:
            db.insert_data(
                model="whisper-1",
                n_input_tokens=int(np.ceil(sound_length_in_seconds)),
            )

        return transcript.text


def _np_array_to_wav_in_memory(array: np.ndarray, sample_rate: int):
    """Convert the recorded array to an in-memory wav file."""
    byte_io = io.BytesIO()
    wav.write(byte_io, rate=sample_rate, data=array)
    byte_io.seek(44)  # Skip the WAV header
    return byte_io.read()


def mp3_ro_wav(mp3_buffer: io.BytesIO, volume_increase_db: int = 0):
    """Convert an mp3 buffer to a wav buffer."""
    mp3_buffer.seek(0)
    sound = pydub.AudioSegment.from_mp3(mp3_buffer)
    if volume_increase_db:
        sound += volume_increase_db

    wav_buffer = io.BytesIO()
    sound.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    return wav_buffer


def _get_lower_alphanumeric(string: str):
    """Return a string with only lowercase alphanumeric characters."""
    return re.sub("[^0-9a-zA-Z]+", " ", string.strip().lower())


def _np_array_to_wav_bytes(array: np.ndarray, sample_rate: int):
    """Convert the recorded array to an in-memory wav file."""
    byte_io = io.BytesIO()
    wav.write(byte_io, rate=sample_rate, data=array)
    byte_io.seek(44)  # Skip the WAV header
    return byte_io.read()


def _get_lower_alphanumeric(string: str):
    """Return a string with only lowercase alphanumeric characters."""
    return re.sub("[^0-9a-zA-Z]+", " ", string.strip().lower())


def bytestring_to_wav_buffer(self, bytestring):
    """Convert a bytestring to a wav buffer."""
    import wave

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.setframerate(self.sample_rate)
        wav_file.writeframes(bytestring)

    wav_buffer.seek(0)  # Reset the buffer position to the beginning
    return wav_buffer


def pygame_sound_to_wav(sound, frame_rate):
    """Convert a Pygame mixer sound to a wav buffer."""
    import wave

    # Get the raw audio data from the Pygame mixer sound
    raw_data = sound.get_raw()

    # Convert the raw data to a bytes array
    audio_data = io.BytesIO(raw_data)

    output_file = io.BytesIO()
    # Open a WAV file and write the audio data
    with wave.open(output_file, "w") as wav_file:
        wav_file.setnchannels(1)  # Set the number of channels (1 for mono, 2 for stereo)
        wav_file.setsampwidth(2)  # Set the sample width in bytes (2 for 16-bit audio)
        wav_file.setframerate(frame_rate)  # Set the sample rate
        wav_file.writeframesraw(audio_data.getbuffer())

    output_file.seek(0)  # Reset the buffer position to the beginning
    return output_file


def subtract_similar_data(signal1, signal_to_be_removed):
    """Subtract similar data from two signals."""
    logger.error("SUBTRACTING SIGNAL: {}, {}", len(signal1), len(signal_to_be_removed))
    if len(signal1) > len(signal_to_be_removed):
        signal1, signal_to_be_removed = signal_to_be_removed, signal1
    logger.error("SUBTRACTING SIGNAL: {}, {}", len(signal1), len(signal_to_be_removed))

    # Pad the shorter signal with zeros to match the length of the longer signal
    len_diff = len(signal_to_be_removed) - len(signal1)
    signal1_padded = np.pad(signal1, (0, len_diff), "constant")

    # Compute cross-correlation
    import scipy

    cross_correlation = scipy.signal.correlate(
        signal1_padded, signal_to_be_removed, mode="full"
    )

    # Find the time offset
    offset = np.argmax(cross_correlation) - len(signal1_padded) + 1

    # Adjust the offset to make it non-negative
    offset = max(0, offset)

    # Subtract similar data
    result_signal = (
        signal1_padded[offset:] - signal_to_be_removed[: len(signal1_padded) - offset]
    )

    logger.error("DONE SUBTRACTING SIGNAL: {}", len(result_signal))

    return result_signal
