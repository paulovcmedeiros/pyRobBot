"""Functions for converting text to speech and speech to text."""
import io
from dataclasses import dataclass

import numpy as np
import pygame
import scipy.io.wavfile as wav
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
from loguru import logger
from pygame import mixer


@dataclass
class LiveAssistant:
    """Class for converting text to speech and speech to text."""

    language: str = "en"
    recording_duration_seconds: int = 5

    def __post_init__(self):
        mixer.init()

    def speak(self, text):
        """Convert text to speech."""
        # Initialize gTTS with the text to convert
        speech = gTTS(text, lang=self.language)

        # Save the audio file to a temporary file
        speech_file = "speech.mp3"
        speech.save(speech_file)

        # Play the audio file
        speech = mixer.Sound(speech_file)
        channel = speech.play()
        while channel.get_busy():
            pygame.time.wait(100)

    def listen(self):
        """Record audio from the microphone and convert it to text."""
        # Record audio from the microphone
        sample_rate = 44100  # Hz
        n_frames = int(self.recording_duration_seconds * sample_rate)
        rec_as_array = sd.rec(
            frames=n_frames, samplerate=sample_rate, channels=1, dtype="int16"
        )
        logger.info("Recording Audio")
        sd.wait()
        logger.info("Done Recording")

        # Convert the recorded array to an in-memory wav file
        byte_io = io.BytesIO()
        wav.write(byte_io, rate=sample_rate, data=rec_as_array.astype(np.int16))
        byte_io.seek(0)  # Reset the file pointer to the beginning of the file

        # Use SpeechRecognition to convert the audio to text
        r = sr.Recognizer()
        with sr.AudioFile(byte_io) as source:
            audio_data = r.listen(source)

        try:
            text = r.recognize_google(audio_data, language=self.language)
        except sr.exceptions.UnknownValueError:
            logger.warning("Could not understand audio")
            text = ""

        return text
