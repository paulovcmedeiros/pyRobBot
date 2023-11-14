"""Functions for converting text to speech and speech to text."""
import io
import queue
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pygame
import scipy.io.wavfile as wav
import soundfile as sf
import speech_recognition as sr
import webrtcvad
from gtts import gTTS
from loguru import logger
from pygame import mixer

try:
    import sounddevice as sd

    _sounddevice_imported = True
except OSError as error:
    logger.error(error)
    logger.error(
        "Can't use module `sounddevice`. Please check your system's PortAudio install."
    )
    _sounddevice_imported = False


@dataclass
class LiveAssistant:
    """Class for converting text to speech and speech to text."""

    language: str = "en"
    inactivity_timeout_seconds: int = 2
    sample_rate: int = 32000  # Hz
    recording_duration_seconds: int = 5
    inactivity_sound_intensity_threshold: float = 0.02

    def __post_init__(self):
        if not _sounddevice_imported:
            logger.error(
                "Module `sounddevice`, needed for audio recording, is not available."
            )
            logger.error("Cannot continue. Exiting.")
            raise SystemExit(1)
        mixer.init()
        self.vad = webrtcvad.Vad(2)

    def speak(self, text):
        """Convert text to speech."""
        logger.debug("Converting text to speech...")
        # Initialize gTTS with the text to convert
        speech = gTTS(text, lang=self.language)

        # Convert the recorded array to an in-memory wav file
        byte_io = io.BytesIO()
        speech.write_to_fp(byte_io)
        byte_io.seek(0)

        logger.debug("Done converting text to speech.")

        # Play the audio file
        speech = mixer.Sound(byte_io)
        channel = speech.play()
        while channel.get_busy():
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

        # From webrtcvad docs: A frame must be either 10, 20, or 30 ms in duration
        frame_duration = 30  # milliseconds
        stream_block_size = int((self.sample_rate * frame_duration) / 1000)
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
                maxlen=int((1000.0 * self.inactivity_timeout_seconds) / frame_duration)
            )
            last_inactivity_checked = datetime.now()
            user_is_speaking = True
            while user_is_speaking:
                new_data = q.get()
                audio_file.write(new_data)

                # Gather voice activity samples for the inactivity check
                is_speech = self.vad.is_speech(
                    _np_array_to_wav_in_memory(new_data, sample_rate=self.sample_rate),
                    self.sample_rate,
                )
                voice_activity_detected.append(is_speech)

                # Decide if user has been inactive for too long
                now = datetime.now()
                if (
                    now - last_inactivity_checked
                ).seconds >= self.inactivity_timeout_seconds:
                    last_inactivity_checked = now
                    user_is_speaking = any(voice_activity_detected)

        # Detect if there was any sound at all, skip the conversion if not
        recorded_audio = np.frombuffer(raw_buffer.getvalue(), dtype=np.int16)
        max_intensity = np.max(np.absolute(recorded_audio))
        if max_intensity < self.inactivity_sound_intensity_threshold:
            logger.debug("No sound detected")
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
