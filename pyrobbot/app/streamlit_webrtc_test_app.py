# https://github.com/whitphx/streamlit-webrtc/blob/main/pages/10_sendonly_audio.py
import queue
import threading
from collections import deque

import av
import pygame
from loguru import logger
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from pyrobbot.voice_chat import VoiceChat

# See
# <https://github.com/whitphx/streamlit-webrtc/blob/
# caf429e858fcf6eaf87096bfbb41be7e269cf0c0/streamlit_webrtc/mix.py#L33>
# and
# <https://github.com/whitphx/streamlit-webrtc/issues/361#issuecomment-894230158>
# for streamlit_webrtc hard-coded value for sample_rate = 48000


chat = VoiceChat()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


received_audio_frames_queue = queue.Queue()
possible_speech_chunks_queue = queue.Queue()
audio_playing_chunks_queue = queue.Queue()
audio_playing_queue = queue.Queue()


def trim_beginning(audio: AudioSegment, **kwargs):
    """Trim the beginning of the audio segment to remove silence."""
    beginning = detect_leading_silence(audio, **kwargs)
    return audio[beginning:]


def trim_ending(audio: AudioSegment, **kwargs):
    """Trim the ending of the audio segments to remove silence."""
    audio = trim_beginning(audio.reverse(), **kwargs)
    return audio.reverse()


def trim_silence(audio: AudioSegment, **kwargs):
    """Trim the silence from the beginning and ending of the audio segment."""
    kwargs["silence_threshold"] = kwargs.get("silence_threshold", -40.0)
    audio = trim_beginning(audio, **kwargs)
    return trim_ending(audio, **kwargs)


def audio_frame_callback(frame: av.AudioFrame):
    """Callback function to receive audio frames from the browser."""
    try:
        # Prevent recording while playing audio
        audio_playing_queue.join()

        logger.info("Received audio frame form stream. Sending to queue.")
        received_audio_frames_queue.put(frame)

        if not possible_speech_chunks_queue.empty():
            new_audio_chunk = possible_speech_chunks_queue.get()
            if new_audio_chunk is None:
                # User has stopped speaking. Concatenate all audios from
                # play_audio_queue and send the result to be played
                concatenated_audio = AudioSegment.empty()
                while not audio_playing_chunks_queue.empty():
                    concatenated_audio += audio_playing_chunks_queue.get()
                audio_playing_queue.put(concatenated_audio)
            else:
                audio_playing_chunks_queue.put(new_audio_chunk)
            possible_speech_chunks_queue.task_done()
    except Exception as error:  # noqa: BLE001
        logger.error(error)


def handle_audio_playing(audio_playing_queue):
    """Play audio."""
    while True:
        audio = trim_silence(audio_playing_queue.get())
        logger.info("Playing audio ({} seconds)", audio.duration_seconds)
        chat.mixer.Sound(audio.raw_data).play()
        while chat.mixer.get_busy():
            pygame.time.wait(100)
        audio_playing_queue.task_done()


audio_playing_thread = threading.Thread(
    target=handle_audio_playing, args=(audio_playing_queue,), daemon=True
)


def run_app():
    """Use WebRTC to transfer audio frames from the browser to the server."""
    webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
        desired_playing_state=True,
        audio_frame_callback=audio_frame_callback,
    )
    audio_playing_thread.start()

    # This deque will be used in order to keep a moving window of audio chunks to monitor
    # voice activity. The length of the deque is calculated such that the concatenated
    # audio chunks will produce an audio at most inactivity_timeout_seconds long
    audio_chunks_moving_window = deque(
        maxlen=int((1000.0 * chat.inactivity_timeout_seconds) / chat.frame_duration)
    )
    moving_window_speech_likelihood = 0.0

    user_has_been_speaking = False
    while True:
        # Receive a new audio frame from the stream
        received_audio_frame = received_audio_frames_queue.get()
        if received_audio_frame.sample_rate != chat.sample_rate:
            raise ValueError(
                f"audio_frame.sample_rate = {received_audio_frame.sample_rate} != "
                f"chat.sample_rate = {chat.sample_rate}"
            )

        # Convert the received audio frame to an AudioSegment object
        raw_samples = received_audio_frame.to_ndarray()
        audio_chunk = AudioSegment(
            data=raw_samples.tobytes(),
            sample_width=received_audio_frame.format.bytes,
            frame_rate=received_audio_frame.sample_rate,
            channels=len(received_audio_frame.layout.channels),
        )
        if audio_chunk.duration_seconds != chat.frame_duration / 1000:
            raise ValueError(
                f"sound_chunk.duration_seconds = {audio_chunk.duration_seconds} != "
                f"chat.frame_duration / 1000 = {chat.frame_duration / 1000}"
            )

        # Resample the AudioSegment to be compatible with the VAD engine
        audio_chunk = audio_chunk.set_frame_rate(chat.sample_rate).set_channels(1)

        # Now do the VAD
        # Check if the current sound chunk is likely to be speech
        vad_thinks_this_chunk_is_speech = chat.vad.is_speech(
            audio_chunk.raw_data, chat.sample_rate
        )

        # Monitor voice activity within moving window of length inactivity_timeout_seconds
        audio_chunks_moving_window.append(
            {"audio": audio_chunk, "is_speech": vad_thinks_this_chunk_is_speech}
        )
        moving_window_length = len(audio_chunks_moving_window)
        if moving_window_length == audio_chunks_moving_window.maxlen:
            voice_activity = (chunk["is_speech"] for chunk in audio_chunks_moving_window)
            moving_window_speech_likelihood = sum(voice_activity) / moving_window_length

        user_speaking_now = (
            moving_window_speech_likelihood >= chat.speech_likelihood_threshold
        )
        if user_has_been_speaking:
            if user_speaking_now:
                possible_speech_chunks_queue.put(audio_chunk)
            else:
                logger.info("User has stopped speaking.")
                user_has_been_speaking = False
                possible_speech_chunks_queue.put(None)
        elif user_speaking_now:
            logger.info("User has started speaking.")
            user_has_been_speaking = True
            for audio_chunk in audio_chunks_moving_window:
                possible_speech_chunks_queue.put(audio_chunk["audio"])

        received_audio_frames_queue.task_done()


if __name__ == "__main__":
    run_app()
