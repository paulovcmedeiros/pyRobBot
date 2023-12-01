"""Code related to the voice chat feature."""
import contextlib
import io
import queue
import re
import socket
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
from pydub import AudioSegment

from .chat import Chat
from .chat_configs import VoiceChatConfigs
from .general_utils import retry

try:
    import sounddevice as sd
except OSError as error:
    logger.error(error)
    logger.error(
        "Can't use module `sounddevice`. Please check your system's PortAudio install."
    )
    _sounddevice_imported = False
else:
    _sounddevice_imported = True

try:
    # Test if pydub's AudioSegment can be used
    with contextlib.suppress(pydub.exceptions.CouldntDecodeError):
        AudioSegment.from_mp3(io.BytesIO())
except (ImportError, OSError, FileNotFoundError) as error:
    logger.error(
        "{}. Can't use module `pydub`. Please check your system's ffmpeg install.", error
    )
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

        if not _pydub_imported:
            raise ImportError(
                "Module `pydub`, needed for audio conversion, is not available."
            )

        super().__init__(configs=configs)

        self.block_size = int((self.sample_rate * self.frame_duration) / 1000)

        self.mixer = pygame.mixer
        self.mixer.init(frequency=self.sample_rate, channels=1, buffer=self.block_size)

        self.recogniser = sr.Recognizer()
        self.recogniser.operation_timeout = self.timeout

        self.vad = webrtcvad.Vad(2)
        chime.theme("big-sur")

        # Create queues for TTS processing and speech playing
        self.questions_queue = queue.Queue()
        self.tts_conversion_queue = queue.Queue()
        self.play_speech_queue = queue.Queue()

        # Create threads to watch the TTS and speech playing queues
        self.questions_queue_watcher_thread = threading.Thread(
            target=self.handle_questions_queue, args=(self.questions_queue,), daemon=True
        )
        self.tts_conversion_watcher_thread = threading.Thread(
            target=self.handle_tts_queue, args=(self.tts_conversion_queue,), daemon=True
        )
        self.play_speech_thread = threading.Thread(
            target=self.handle_speak_queue, args=(self.play_speech_queue,), daemon=True
        )

    def start(self):
        """Start the chat."""
        # ruff: noqa: T201
        self.tts_conversion_watcher_thread.start()
        self.play_speech_thread.start()

        if not self.skip_initial_greeting:
            self.tts_conversion_queue.put(self.initial_greeting)

        self.questions_queue_watcher_thread.start()

        try:
            while True:
                self.tts_conversion_queue.join()
                self.play_speech_queue.join()
                chime.warning()
                logger.debug(f"{self.assistant_name}> Listening...")

                question = self.questions_queue.get()
                if question is None:
                    raise InterruptedError

                chime.success()
                self.answer_question(question)
                self.questions_queue.task_done()

        except (KeyboardInterrupt, EOFError):
            chime.info()
        except InterruptedError:
            chime.theme("material")
            chime.error()
        finally:
            logger.debug("Leaving chat")

    def answer_question(self, question: str):
        """Answer a question."""
        logger.debug("{}> Getting response to '{}'...", self.assistant_name, question)
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

    def speak(self, audio: AudioSegment):
        """Reproduce audio from a pygame Sound object."""
        self.mixer.Sound(file=audio.export()).play()
        while self.mixer.get_busy():
            pygame.time.wait(100)

    def listen(self) -> AudioSegment:
        """Record audio from the microphone until user stops."""
        # Adapted from
        # <https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html#
        #  recording-with-arbitrary-duration>
        logger.debug("The assistant is listening...")
        q = queue.Queue()

        def callback(indata, frames, time, status):  # noqa: ARG001
            """This is called (from a separate thread) for each audio block."""
            q.put(indata.copy())

        raw_buffer = io.BytesIO()
        with self.get_sound_file(raw_buffer, mode="x") as sound_file, sd.InputStream(
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
                    sound_file.write(new_data)

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
            return AudioSegment.empty()

        return AudioSegment.from_wav(self.trim_audio(raw_buffer))

    def stt(self, audio_segment: AudioSegment) -> str:
        """Perform speech-to-text: transcribe text from an AudioData obj."""
        if not audio_segment:
            logger.debug("No speech detected")
            return ""

        if self.stt_engine == "openai":
            stt_function = self._stt_openai
            fallback_stt_function = self._stt_google
            fallback_name = "google"
        else:
            stt_function = self._stt_google
            fallback_stt_function = self._stt_openai
            fallback_name = "openai"

        logger.debug("Converting audio to text ({} STT)...", self.stt_engine)
        try:
            rtn = stt_function(audio_segment)
        except (ConnectionResetError, socket.timeout) as error:
            logger.error(error)
            logger.error(
                "Can't communicate with `{}` speech-to-text API right now",
                self.stt_engine,
            )
            logger.warning("Trying to use `{}` STT instead", fallback_name)
            rtn = fallback_stt_function(audio_segment)
        except sr.exceptions.UnknownValueError:
            rtn = ""

        return rtn.strip()

    def handle_questions_queue(self, questions_queue: queue.Queue):
        """Handle the queue of questions to be answered."""
        minimum_question_duration_seconds = 0.1
        while True:
            try:
                if self._assistant_still_replying():
                    continue

                audio = self.listen()
                if audio.duration_seconds < minimum_question_duration_seconds:
                    continue
                question = self.stt(audio).strip()

                # Check for the exit expressions
                if any(
                    _get_lower_alphanumeric(question).startswith(
                        _get_lower_alphanumeric(expr)
                    )
                    for expr in self.exit_expressions
                ):
                    questions_queue.put(None)
                elif question:
                    questions_queue.put(question)
            except Exception as error:  # noqa: BLE001
                logger.exception(error)

    def handle_speak_queue(self, audio_queue: queue.Queue[AudioSegment]):
        """Handle the queue of audio segments to be played."""
        while True:
            try:
                self.speak(audio_queue.get())
            except Exception as error:  # noqa: PERF203, BLE001
                logger.exception(error)
            finally:
                audio_queue.task_done()

    def handle_tts_queue(self, text_queue: queue.Queue):
        """Handle the text-to-speech queue."""
        while True:
            try:
                text = text_queue.get()
                logger.debug("Received for {} TTS: '{}'", self.tts_engine, text)

                if self.tts_engine == "openai":
                    tts_audio_segment = self._tts_openai(text)
                else:
                    tts_audio_segment = self._tts_google(text)

                # Dispatch the audio to be played
                self.play_speech_queue.put(tts_audio_segment)

                logger.debug("Done with TTS for '{}'", text)
            except Exception as error:  # noqa: PERF203, BLE001
                logger.opt(exception=True).debug(error)
                logger.error(error)
            finally:
                text_queue.task_done()

    def trim_audio(self, raw_buffer: io.BytesIO):
        """Trim the audio data to remove silence from the beginning and end."""
        raw_buffer.seek(0)
        with sr.AudioFile(raw_buffer) as source:
            audio_data = self.recogniser.listen(source)

        return io.BytesIO(audio_data.get_wav_data())

    def get_sound_file(self, wav_buffer: io.BytesIO, mode: str = "r"):
        """Return a sound file object."""
        return sf.SoundFile(
            wav_buffer,
            mode=mode,
            samplerate=self.sample_rate,
            channels=1,
            format="wav",
            subtype="PCM_16",
        )

    def _assistant_still_replying(self):
        """Check if the assistant is still talking."""
        return (
            self.mixer.get_busy()
            or self.questions_queue.unfinished_tasks > 0
            or self.tts_conversion_queue.unfinished_tasks > 0
            or self.play_speech_queue.unfinished_tasks > 0
        )

    def _tts_openai(self, text) -> AudioSegment:
        """Convert text to speech using OpenAI's TTS. Return an AudioSegment object."""
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

        audio = AudioSegment.from_mp3(mp3_buffer)
        audio += 8  # Increase volume by 8 dB
        return audio

    def _tts_google(self, text) -> AudioSegment:
        """Convert text to speech using Google's TTS. Return a WAV BytesIO object."""
        tts = gTTS(text.strip(), lang=self.language)
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        return AudioSegment.from_mp3(mp3_buffer)

    @retry()
    def _stt_openai(self, audio_segment: AudioSegment):
        """Perform speech-to-text using OpenAI's API."""
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.name = "audio.wav"
        wav_buffer.seek(0)
        with wav_buffer as audio_file_buffer:
            transcript = OpenAI(timeout=self.timeout).audio.transcriptions.create(
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
                n_input_tokens=int(np.ceil(audio_segment.duration_seconds)),
            )

        return transcript.text

    def _stt_google(self, audio_segment: AudioSegment):
        """Perform speech-to-text using Google's API."""
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        with sr.AudioFile(wav_buffer) as source:
            audio_data = self.recogniser.record(source)

        return self.recogniser.recognize_google(
            audio_data=audio_data, language=self.language
        )


def _np_array_to_wav_in_memory(array: np.ndarray, sample_rate: int):
    """Convert the recorded array to an in-memory wav file."""
    byte_io = io.BytesIO()
    wav.write(byte_io, rate=sample_rate, data=array)
    byte_io.seek(44)  # Skip the WAV header
    return byte_io.read()


def _get_lower_alphanumeric(string: str):
    """Return a string with only lowercase alphanumeric characters."""
    return re.sub("[^0-9a-zA-Z]+", " ", string.strip().lower())


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
