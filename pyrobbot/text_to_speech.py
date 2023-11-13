"""Functions for converting text to speech and speech to text."""
import io
import queue
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pygame
import scipy.io.wavfile as wav
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
from loguru import logger
from pygame import mixer


@dataclass
class LiveAssistant:
    """Class for converting text to speech and speech to text."""

    language: str = "en"
    recording_duration_seconds: int = 5
    inactivity_timeout_seconds: int = 2
    inactivity_sound_intensity_threshold: float = 0.015

    def __post_init__(self):
        mixer.init()

    def speak(self, text):
        """Convert text to speech."""
        logger.info("Converting text to speech...")
        # Initialize gTTS with the text to convert
        speech = gTTS(text, lang=self.language)

        # Convert the recorded array to an in-memory wav file
        byte_io = io.BytesIO()
        speech.write_to_fp(byte_io)
        byte_io.seek(0)

        logger.info("Done converting text to speech.")

        # Play the audio file
        speech = mixer.Sound(byte_io)
        channel = speech.play()
        while channel.get_busy():
            pygame.time.wait(100)

    def listen_time_limited(self):
        """Record audio from the mic, for a limited timelength, and convert it to text."""
        sample_rate = 44100  # Hz
        n_frames = int(self.recording_duration_seconds * sample_rate)
        # Record audio from the microphone
        rec_as_array = sd.rec(
            frames=n_frames, samplerate=sample_rate, channels=2, dtype="int16"
        )
        logger.info("Recording Audio")
        sd.wait()
        logger.info("Done Recording")

        logger.info("Converting audio to text...")
        # Convert the recorded array to an in-memory wav file
        byte_io = io.BytesIO()
        wav.write(byte_io, rate=sample_rate, data=rec_as_array.astype(np.int16))
        text = self._audio_buffer_to_text(self, byte_io)
        logger.info("Done converting audio to text.")

        return text

    def listen(self):
        """Record audio from the microphone, until user stops, and convert it to text."""
        q = queue.Queue()

        def callback(indata, frames, time, status):  # noqa: ARG001
            """This is called (from a separate thread) for each audio block."""
            q.put(indata.copy())

        raw_buffer = io.BytesIO()
        with sf.SoundFile(
            raw_buffer,
            mode="x",
            samplerate=44100,
            channels=2,
            format="wav",
            subtype="PCM_16",
        ) as audio_file, sd.InputStream(samplerate=44100, channels=2, callback=callback):
            # Recording will stop after self.inactivity_timeout_seconds of silence
            max_intensity = 1.0
            last_checked = datetime.now()
            while max_intensity > self.inactivity_sound_intensity_threshold:
                new_data = q.get()
                audio_file.write(new_data)
                now = datetime.now()
                if (now - last_checked).seconds > self.inactivity_timeout_seconds:
                    last_checked = now
                    max_intensity = np.max([abs(np.min(new_data)), abs(np.max(new_data))])

        logger.info("Converting audio to text...")
        text = self._audio_buffer_to_text(byte_io=raw_buffer)
        logger.info("Done converting audio to text.")

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
            logger.warning("Could not understand audio")
            return ""
