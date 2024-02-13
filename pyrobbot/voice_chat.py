"""Code related to the voice chat feature."""
import contextlib
import io
import queue
import threading
import time
from collections import deque
from datetime import datetime

import chime
import numpy as np
import pydub
import pygame
import soundfile as sf
import webrtcvad
from loguru import logger
from pydub import AudioSegment

from .chat import Chat
from .chat_configs import VoiceChatConfigs
from .general_utils import _get_lower_alphanumeric, str2_minus_str1
from .sst_and_tts import SpeechToText, TextToSpeech

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
        _check_needed_imports()
        super().__init__(configs=configs)

        self.block_size = int((self.sample_rate * self.frame_duration) / 1000)

        self.mixer = pygame.mixer
        self.mixer.init(frequency=self.sample_rate, channels=1, buffer=self.block_size)

        self.vad = webrtcvad.Vad(2)

        self.default_chime_theme = "big-sur"
        chime.theme(self.default_chime_theme)

        # Create queues and threads for handling the chat
        # 1. Watching for questions from the user
        self.questions_queue = queue.Queue()
        self.questions_listening_watcher_thread = threading.Thread(
            target=self.handle_question_listening,
            args=(self.questions_queue,),
            daemon=True,
        )
        # 2. Converting assistant's text reply to speech and playing it
        self.tts_conversion_queue = queue.Queue()
        self.play_speech_queue = queue.Queue()
        self.tts_conversion_watcher_thread = threading.Thread(
            target=self.handle_tts_queue, args=(self.tts_conversion_queue,), daemon=True
        )
        self.play_speech_thread = threading.Thread(
            target=self.handle_speech_queue, args=(self.play_speech_queue,), daemon=True
        )
        # 3. Watching for expressions that cancel the reply or exit the chat
        self.check_for_interrupt_expressions_queue = queue.Queue()
        self.check_for_interrupt_expressions_thread = threading.Thread(
            target=self.check_for_interrupt_expressions_handler,
            args=(self.check_for_interrupt_expressions_queue,),
            daemon=True,
        )
        self.interrupt_reply = threading.Event()
        self.exit_chat = threading.Event()

    def start(self):
        """Start the chat."""
        # ruff: noqa: T201
        self.tts_conversion_watcher_thread.start()
        self.play_speech_thread.start()
        if not self.skip_initial_greeting:
            self.tts_conversion_queue.put(self.initial_greeting)
            while self._assistant_still_replying():
                pygame.time.wait(50)
        self.questions_listening_watcher_thread.start()
        self.check_for_interrupt_expressions_thread.start()

        while not self.exit_chat.is_set():
            try:
                self.tts_conversion_queue.join()
                self.play_speech_queue.join()

                if self.interrupt_reply.is_set():
                    logger.opt(colors=True).debug(
                        "<yellow>Interrupting the reply</yellow>"
                    )
                    self.interrupt_reply.clear()
                    with self.check_for_interrupt_expressions_queue.mutex:
                        self.check_for_interrupt_expressions_queue.queue.clear()
                    with contextlib.suppress(pygame.error):
                        self.mixer.stop()
                    with self.questions_queue.mutex:
                        self.questions_queue.queue.clear()
                    chime.theme("material")
                    chime.error()
                    chime.theme(self.default_chime_theme)
                    time.sleep(0.25)

                chime.warning()
                logger.debug(f"{self.assistant_name}> Waiting for user input...")
                question = self.questions_queue.get()
                self.questions_queue.task_done()
                if question is None:
                    raise EOFError

                chime.success()
                self.answer_question(question)

            except (KeyboardInterrupt, EOFError):  # noqa: PERF203
                chime.info()
                self.exit_chat.set()

        logger.debug("Leaving chat")

    def answer_question(self, question: str):
        """Answer a question."""
        logger.debug("{}> Getting response to '{}'...", self.assistant_name, question)
        sentence = ""
        inside_code_block = False
        at_least_one_code_line_written = False
        for answer_chunk in self.respond_user_prompt(prompt=question):
            if self.interrupt_reply.is_set() or self.exit_chat.is_set():
                return

            fmtd_chunk = answer_chunk.strip(" \n")
            code_block_start_detected = fmtd_chunk.startswith("``")

            if code_block_start_detected and not inside_code_block:
                # Toggle the code block state
                inside_code_block = True

            if inside_code_block:
                code_chunk = answer_chunk
                if at_least_one_code_line_written:
                    inside_code_block = not fmtd_chunk.endswith("``")  # Code block ends
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
                stripd_chunk = answer_chunk.strip()
                if stripd_chunk.endswith(("?", "!", ".")):
                    # Check if second last character is a number, to avoid splitting
                    if stripd_chunk.endswith("."):
                        with contextlib.suppress(IndexError):
                            previous_char = sentence.strip()[-2]
                            if previous_char.isdigit():
                                continue
                    # Send sentence for TTS even if the request hasn't finished
                    self.tts_conversion_queue.put(sentence)
                    sentence = ""
        if sentence:
            self.tts_conversion_queue.put(sentence)
        if at_least_one_code_line_written:
            spoken_info_to_user = "The code has been written to the console"
            spoken_info_to_user = self._translate(spoken_info_to_user)
            self.tts_conversion_queue.put(spoken_info_to_user)

    def speak(self, tts: TextToSpeech):
        """Reproduce audio from a pygame Sound object."""
        tts.set_sample_rate(self.sample_rate)
        channel = self.mixer.Sound(tts.speech.raw_data).play()
        audio_recorded_while_assistant_replies = self.listen(
            duration_seconds=tts.speech.duration_seconds
        )
        while channel.get_busy():
            pygame.time.wait(100)

        msgs_to_compare = {
            "assistant_txt": tts.text,
            "user_audio": audio_recorded_while_assistant_replies,
        }
        self.check_for_interrupt_expressions_queue.put(msgs_to_compare)

    def check_for_interrupt_expressions_handler(
        self, check_for_interrupt_expressions_queue: queue.Queue
    ):
        """Check for expressions that interrupt the assistant's reply."""
        while not self.exit_chat.is_set():
            try:
                msgs_to_compare = check_for_interrupt_expressions_queue.get()
                assistant_msg = _get_lower_alphanumeric(msgs_to_compare["assistant_txt"])
                recorded_prompt = SpeechToText(
                    speech=msgs_to_compare["user_audio"],
                    engine=self.stt_engine,
                    language=self.language,
                    timeout=self.timeout,
                    general_token_usage_db=self.general_token_usage_db,
                    token_usage_db=self.token_usage_db,
                ).text
                recorded_prompt = _get_lower_alphanumeric(recorded_prompt)

                user_words = str2_minus_str1(
                    str1=assistant_msg.strip(), str2=recorded_prompt.strip()
                ).strip()
                if user_words:
                    logger.debug(
                        "Detected user words while assistant was speaking: {}",
                        user_words,
                    )
                    if any(
                        cancel_cmd in user_words for cancel_cmd in self.cancel_expressions
                    ):
                        logger.debug(
                            "Heard '{}'. Signalling for reply to be cancelled...",
                            user_words,
                        )
                        self.interrupt_reply.set()
            except Exception as error:  # noqa: PERF203, BLE001
                logger.opt(exception=True).debug(error)
            finally:
                check_for_interrupt_expressions_queue.task_done()

    def listen(self, duration_seconds: float = np.inf) -> AudioSegment:
        """Record audio from the microphone until user stops."""
        # Adapted from
        # <https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html#
        #  recording-with-arbitrary-duration>
        logger.debug(
            "The assistant is listening for (in principle) {} s...", duration_seconds
        )
        q = queue.Queue()

        def callback(indata, frames, time, status):  # noqa: ARG001
            """This is called (from a separate thread) for each audio block."""
            q.put(indata.copy())

        raw_buffer = io.BytesIO()
        start_time = datetime.now()
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
            elapsed_time = 0.0
            with contextlib.suppress(KeyboardInterrupt):
                while user_is_speaking and elapsed_time < duration_seconds:
                    new_data = q.get()
                    sound_file.write(new_data)

                    # Gather voice activity samples for the inactivity check
                    wav_buffer = _np_array_to_wav_in_memory(
                        sound_data=new_data,
                        sample_rate=self.sample_rate,
                        subtype="PCM_16",
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

                    elapsed_time = (now - start_time).seconds

        if speech_detected:
            return AudioSegment.from_wav(raw_buffer)
        return AudioSegment.empty()

    def handle_question_listening(self, questions_queue: queue.Queue):
        """Handle the queue of questions to be answered."""
        minimum_prompt_duration_seconds = 0.1
        while not self.exit_chat.is_set():
            try:
                if self._assistant_still_replying():
                    continue

                audio = self.listen()
                if audio is None:
                    questions_queue.put(None)
                    continue
                if audio.duration_seconds < minimum_prompt_duration_seconds:
                    continue

                question = SpeechToText(
                    speech=audio,
                    engine=self.stt_engine,
                    language=self.language,
                    timeout=self.timeout,
                    general_token_usage_db=self.general_token_usage_db,
                    token_usage_db=self.token_usage_db,
                ).text

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
            except sd.PortAudioError as error:
                logger.opt(exception=True).debug(error)
            except Exception as error:  # noqa: BLE001
                logger.opt(exception=True).debug(error)
                logger.error(error)

    def handle_speech_queue(self, speech_queue: queue.Queue[TextToSpeech]):
        """Handle the queue of audio segments to be played."""
        while not self.exit_chat.is_set():
            try:
                speech = speech_queue.get()
                if speech and not self.interrupt_reply.is_set():
                    self.speak(speech)
            except Exception as error:  # noqa: BLE001, PERF203
                logger.exception(error)
            finally:
                speech_queue.task_done()

    def handle_tts_queue(self, text_queue: queue.Queue):
        """Handle the text-to-speech queue."""
        while not self.exit_chat.is_set():
            try:
                text = text_queue.get()
                if text and not self.interrupt_reply.is_set():
                    tts = TextToSpeech(
                        text=text,
                        engine=self.tts_engine,
                        openai_tts_voice=self.openai_tts_voice,
                        language=self.language,
                        timeout=self.timeout,
                        general_token_usage_db=self.general_token_usage_db,
                        token_usage_db=self.token_usage_db,
                    )

                    # Trigger the TTS conversion
                    _ = tts.speech

                    # Dispatch the audio to be played
                    self.play_speech_queue.put(tts)
            except Exception as error:  # noqa: PERF203, BLE001
                logger.opt(exception=True).debug(error)
                logger.error(error)
            finally:
                text_queue.task_done()

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


def _check_needed_imports():
    """Check if the needed modules are available."""
    if not _sounddevice_imported:
        raise ImportError(
            "Module `sounddevice`, needed for audio recording, is not available."
        )

    if not _pydub_imported:
        raise ImportError(
            "Module `pydub`, needed for audio conversion, is not available."
        )


def _np_array_to_wav_in_memory(
    sound_data: np.ndarray, sample_rate: int, subtype="PCM_16"
):
    """Convert the recorded array to an in-memory wav file."""
    wav_buffer = io.BytesIO()
    wav_buffer.name = "audio.wav"
    sf.write(wav_buffer, sound_data, sample_rate, subtype=subtype)
    wav_buffer.seek(44)  # Skip the WAV header
    return wav_buffer.read()
