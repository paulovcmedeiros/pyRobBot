"""Code related to the voice chat feature."""

import contextlib
import io
import queue
import threading
import time
from collections import defaultdict, deque
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
from .sst_and_tts import TextToSpeech

try:
    import sounddevice as sd
except OSError as error:
    logger.exception(error)
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
    logger.exception(error)
    logger.error("Can't use module `pydub`. Please check your system's ffmpeg install.")
    _pydub_usable = False
else:
    _pydub_usable = True


class VoiceChat(Chat):
    """Class for converting text to speech and speech to text."""

    default_configs = VoiceChatConfigs()

    def __init__(self, configs: VoiceChatConfigs = default_configs, **kwargs):
        """Initializes a chat instance."""
        super().__init__(configs=configs, **kwargs)
        _check_needed_imports()

        self.block_size = int((self.sample_rate * self.frame_duration) / 1000)

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
            target=self.handle_tts_conversion_queue,
            args=(self.tts_conversion_queue,),
            daemon=True,
        )
        self.play_speech_thread = threading.Thread(
            target=self.handle_play_speech_queue,
            args=(self.play_speech_queue,),
            daemon=True,
        )  # TODO: Do not start this in webchat
        # 3. Watching for expressions that cancel the reply or exit the chat
        self.check_for_interrupt_expressions_queue = queue.Queue()
        self.check_for_interrupt_expressions_thread = threading.Thread(
            target=self.check_for_interrupt_expressions_handler,
            args=(self.check_for_interrupt_expressions_queue,),
            daemon=True,
        )
        self.interrupt_reply = threading.Event()
        self.exit_chat = threading.Event()

        # Keep track of played audios to update the history db
        self.current_answer_audios_queue = queue.Queue()
        self.handle_update_audio_history_thread = threading.Thread(
            target=self.handle_update_audio_history,
            args=(self.current_answer_audios_queue,),
            daemon=True,
        )

    @property
    def mixer(self):
        """Return the mixer object."""
        mixer = getattr(self, "_mixer", None)
        if mixer is not None:
            return mixer

        self._mixer = pygame.mixer
        try:
            self.mixer.init(
                frequency=self.sample_rate, channels=1, buffer=self.block_size
            )
        except pygame.error as error:
            logger.exception(error)
            logger.error(
                "Can't initialize the mixer. Please check your system's audio settings."
            )
            logger.warning("Voice chat may not be available or may not work as expected.")
        return self._mixer

    def start(self):
        """Start the chat."""
        # ruff: noqa: T201
        self.tts_conversion_watcher_thread.start()
        self.play_speech_thread.start()
        if not self.skip_initial_greeting:
            tts_entry = {"exchange_id": self.id, "text": self.initial_greeting}
            self.tts_conversion_queue.put(tts_entry)
            while self._assistant_still_replying():
                pygame.time.wait(50)
        self.questions_listening_watcher_thread.start()
        self.check_for_interrupt_expressions_thread.start()
        self.handle_update_audio_history_thread.start()

        with contextlib.suppress(KeyboardInterrupt, EOFError):
            while not self.exit_chat.is_set():
                self.tts_conversion_queue.join()
                self.play_speech_queue.join()
                self.current_answer_audios_queue.join()

                if self.interrupt_reply.is_set():
                    logger.opt(colors=True).debug(
                        "<yellow>Interrupting the reply</yellow>"
                    )
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
                self.interrupt_reply.clear()
                logger.debug(f"{self.assistant_name}> Waiting for user input...")
                question = self.questions_queue.get()
                self.questions_queue.task_done()

                if question is None:
                    self.exit_chat.set()
                else:
                    chime.success()
                    for chunk in self.answer_question(question):
                        if chunk.chunk_type == "code":
                            print(chunk.content, end="", flush=True)

        self.exit_chat.set()
        chime.info()
        logger.debug("Leaving chat")

    def answer_question(self, question: str):
        """Answer a question."""
        logger.debug("{}> Getting response to '{}'...", self.assistant_name, question)
        sentence_for_tts = ""
        any_code_chunk_yet = False
        for answer_chunk in self.respond_user_prompt(prompt=question):
            if self.interrupt_reply.is_set() or self.exit_chat.is_set():
                logger.debug("Reply interrupted.")
                raise StopIteration
            yield answer_chunk

            if not self.reply_only_as_text:
                if answer_chunk.chunk_type not in ("text", "code"):
                    raise NotImplementedError(
                        "Unexpected chunk type: {}".format(answer_chunk.chunk_type)
                    )

                if answer_chunk.chunk_type == "text":
                    # The answer chunk is to be spoken
                    sentence_for_tts += answer_chunk.content
                    stripd_chunk = answer_chunk.content.strip()
                    if stripd_chunk.endswith(("?", "!", ".")):
                        # Check if second last character is a number, to avoid splitting
                        if stripd_chunk.endswith("."):
                            with contextlib.suppress(IndexError):
                                previous_char = sentence_for_tts.strip()[-2]
                                if previous_char.isdigit():
                                    continue
                        # Send sentence for TTS even if the request hasn't finished
                        tts_entry = {
                            "exchange_id": answer_chunk.exchange_id,
                            "text": sentence_for_tts,
                        }
                        self.tts_conversion_queue.put(tts_entry)
                        sentence_for_tts = ""
                elif answer_chunk.chunk_type == "code" and not any_code_chunk_yet:
                    msg = self._translate("Code will be displayed in the text output.")
                    tts_entry = {"exchange_id": answer_chunk.exchange_id, "text": msg}
                    self.tts_conversion_queue.put(tts_entry)
                    any_code_chunk_yet = True

        if sentence_for_tts and not self.reply_only_as_text:
            tts_entry = {
                "exchange_id": answer_chunk.exchange_id,
                "text": sentence_for_tts,
            }
            self.tts_conversion_queue.put(tts_entry)

        # Signal that the current answer is finished
        tts_entry = {"exchange_id": answer_chunk.exchange_id, "text": None}
        self.tts_conversion_queue.put(tts_entry)

    def handle_update_audio_history(self, current_answer_audios_queue: queue.Queue):
        """Handle updating the chat history with the replies' audio file paths."""
        # Merge all AudioSegments in self.current_answer_audios_queue into a single one
        merged_audios = defaultdict(AudioSegment.empty)
        while not self.exit_chat.is_set():
            try:
                logger.debug("Waiting for reply audio chunks to concatenate and save...")
                audio_chunk_queue_item = current_answer_audios_queue.get()
                reply_audio_chunk = audio_chunk_queue_item["speech"]
                exchange_id = audio_chunk_queue_item["exchange_id"]
                logger.debug("Received audio chunk for response ID {}", exchange_id)

                if reply_audio_chunk is not None:
                    # Reply not yet finished
                    merged_audios[exchange_id] += reply_audio_chunk
                    logger.debug(
                        "Response ID {} audio: {}s so far",
                        exchange_id,
                        merged_audios[exchange_id].duration_seconds,
                    )
                    current_answer_audios_queue.task_done()
                    continue

                # Now the reply has finished
                logger.debug(
                    "Creating a single audio file for response ID {}...", exchange_id
                )
                merged_audio = merged_audios[exchange_id]
                # Update the chat history with the audio file path
                fpath = self.audio_cache_dir() / f"{datetime.now().isoformat()}.mp3"
                logger.debug("Updating chat history with audio file path {}", fpath)
                self.context_handler.database.insert_assistant_audio_file_path(
                    exchange_id=exchange_id, file_path=fpath
                )
                # Save the combined audio as an mp3 file in the cache directory
                merged_audio.export(fpath, format="mp3")
                logger.debug("File {} stored", fpath)
                del merged_audios[exchange_id]
                current_answer_audios_queue.task_done()
            except Exception as error:  # noqa: BLE001
                logger.error(error)
                logger.opt(exception=True).debug(error)

    def speak(self, tts: TextToSpeech):
        """Reproduce audio from a pygame Sound object."""
        tts.set_sample_rate(self.sample_rate)
        self.mixer.Sound(tts.speech.raw_data).play()
        audio_recorded_while_assistant_replies = self.listen(
            duration_seconds=tts.speech.duration_seconds
        )

        msgs_to_compare = {
            "assistant_txt": tts.text,
            "user_audio": audio_recorded_while_assistant_replies,
        }
        self.check_for_interrupt_expressions_queue.put(msgs_to_compare)

        while self.mixer.get_busy():
            pygame.time.wait(100)

    def check_for_interrupt_expressions_handler(
        self, check_for_interrupt_expressions_queue: queue.Queue
    ):
        """Check for expressions that interrupt the assistant's reply."""
        while not self.exit_chat.is_set():
            try:
                msgs_to_compare = check_for_interrupt_expressions_queue.get()
                recorded_prompt = self.stt(speech=msgs_to_compare["user_audio"]).text

                recorded_prompt = _get_lower_alphanumeric(recorded_prompt).strip()
                assistant_msg = _get_lower_alphanumeric(
                    msgs_to_compare.get("assistant_txt", "")
                ).strip()

                user_words = str2_minus_str1(
                    str1=assistant_msg, str2=recorded_prompt
                ).strip()
                if user_words:
                    logger.debug(
                        "Detected user words while assistant was replying: {}",
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
        debug_msg = "The assistant is listening"
        if duration_seconds < np.inf:
            debug_msg += f" for {duration_seconds} s"
        debug_msg += "..."

        inactivity_timeout_seconds = self.inactivity_timeout_seconds
        if duration_seconds < np.inf:
            inactivity_timeout_seconds = duration_seconds

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
            logger.debug("{}", debug_msg)
            # Recording will stop after inactivity_timeout_seconds of silence
            voice_activity_detected = deque(
                maxlen=int((1000.0 * inactivity_timeout_seconds) / self.frame_duration)
            )
            last_inactivity_checked = datetime.now()
            continue_recording = True
            speech_detected = False
            elapsed_time = 0.0
            with contextlib.suppress(KeyboardInterrupt):
                while continue_recording and elapsed_time < duration_seconds:
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
                    if duration_seconds < np.inf:
                        continue_recording = True
                    elif (
                        now - last_inactivity_checked
                    ).seconds >= inactivity_timeout_seconds:
                        speech_likelihood = 0.0
                        if len(voice_activity_detected) > 0:
                            speech_likelihood = sum(voice_activity_detected) / len(
                                voice_activity_detected
                            )
                        continue_recording = (
                            speech_likelihood >= self.speech_likelihood_threshold
                        )
                        if continue_recording:
                            speech_detected = True
                        last_inactivity_checked = now

                    elapsed_time = (now - start_time).seconds

        if speech_detected or duration_seconds < np.inf:
            return AudioSegment.from_wav(raw_buffer)
        return AudioSegment.empty()

    def handle_question_listening(self, questions_queue: queue.Queue):
        """Handle the queue of questions to be answered."""
        minimum_prompt_duration_seconds = 0.05
        while not self.exit_chat.is_set():
            if self._assistant_still_replying():
                pygame.time.wait(100)
                continue
            try:
                audio = self.listen()
                if audio is None:
                    questions_queue.put(None)
                    continue

                if audio.duration_seconds < minimum_prompt_duration_seconds:
                    continue

                question = self.stt(speech=audio).text

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

    def handle_play_speech_queue(self, play_speech_queue: queue.Queue[TextToSpeech]):
        """Handle the queue of audio segments to be played."""
        while not self.exit_chat.is_set():
            try:
                play_speech_queue_item = play_speech_queue.get()
                if play_speech_queue_item["speech"] and not self.interrupt_reply.is_set():
                    self.speak(play_speech_queue_item["tts_obj"])
            except Exception as error:  # noqa: BLE001, PERF203
                logger.exception(error)
            finally:
                play_speech_queue.task_done()

    def handle_tts_conversion_queue(self, tts_conversion_queue: queue.Queue):
        """Handle the text-to-speech queue."""
        logger.debug("Chat {}: TTS conversion handler started.", self.id)
        while not self.exit_chat.is_set():
            try:
                tts_entry = tts_conversion_queue.get()
                if tts_entry["text"] is None:
                    # Signal that the current anwer is finished
                    play_speech_queue_item = {
                        "exchange_id": tts_entry["exchange_id"],
                        "speech": None,
                    }
                    self.play_speech_queue.put(play_speech_queue_item)
                    self.current_answer_audios_queue.put(play_speech_queue_item)

                    logger.debug(
                        "Reply ID {} notified that is has finished",
                        tts_entry["exchange_id"],
                    )
                    tts_conversion_queue.task_done()
                    continue

                text = tts_entry["text"].strip()
                if text and not self.interrupt_reply.is_set():
                    logger.debug(
                        "Reply ID {}: received text '{}' for TTS",
                        tts_entry["exchange_id"],
                        text,
                    )

                    tts_obj = self.tts(text)
                    # Trigger the TTS conversion
                    _ = tts_obj.speech

                    logger.debug(
                        "Reply ID {}: Sending speech for '{}' to the playing queue",
                        tts_entry["exchange_id"],
                        text,
                    )
                    play_speech_queue_item = {
                        "exchange_id": tts_entry["exchange_id"],
                        "tts_obj": tts_obj,
                        "speech": tts_obj.speech,
                    }
                    self.play_speech_queue.put(play_speech_queue_item)
                    self.current_answer_audios_queue.put(play_speech_queue_item)

                # Pay attention to the indentation level
                tts_conversion_queue.task_done()

            except Exception as error:  # noqa: BLE001
                logger.opt(exception=True).debug(error)
                logger.error(error)
        logger.error("TTS conversion queue handler ended.")

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

    def audio_cache_dir(self):
        """Return the audio cache directory."""
        directory = self.cache_dir / "audio_files"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

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
        logger.warning(
            "Module `sounddevice`, needed for local audio recording, is not available."
        )

    if not _pydub_usable:
        logger.error(
            "Module `pydub`, needed for audio conversion, doesn't seem to be working. "
            "Voice chat may not be available or may not work as expected."
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
