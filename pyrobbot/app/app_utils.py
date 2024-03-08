"""Utility functions and classes for the app."""

import contextlib
import datetime
import os
import queue
import threading
from typing import TYPE_CHECKING

import streamlit as st
from loguru import logger
from PIL import Image
from pydub import AudioSegment
from streamlit.runtime.scriptrunner import add_script_run_ctx
from twilio.rest import Client as TwilioClient

from pyrobbot import GeneralDefinitions
from pyrobbot.chat import AssistantResponseChunk
from pyrobbot.voice_chat import VoiceChat

if TYPE_CHECKING:
    from .app_page_templates import AppPage


class WebAppChat(VoiceChat):
    """A chat object for web apps."""

    def __init__(self, **kwargs):
        """Initialize a new instance of the WebAppChat class."""
        super().__init__(**kwargs)
        self.tts_conversion_watcher_thread.start()
        self.handle_update_audio_history_thread.start()


class AsyncReplier:
    """Asynchronously reply to a prompt and stream the text & audio reply."""

    def __init__(self, app_page: "AppPage", prompt: str):
        """Initialize a new instance of the AsyncReplier class."""
        self.app_page = app_page
        self.prompt = prompt

        self.chat_obj = app_page.chat_obj
        self.question_answer_chunks_queue = queue.Queue()

        self.threads = [
            threading.Thread(name="queue_text_chunks", target=self.queue_text_chunks),
            threading.Thread(name="play_queued_audios", target=self.play_queued_audios),
        ]

        self.start()

    def start(self):
        """Start the threads."""
        for thread in self.threads:
            add_script_run_ctx(thread)
            thread.start()

    def join(self):
        """Wait for all threads to finish."""
        logger.debug("Waiting for {} to finish...", type(self).__name__)
        for thread in self.threads:
            thread.join()
        logger.debug("All {} threads finished", type(self).__name__)

    def queue_text_chunks(self):
        """Get chunks of the text reply to the prompt and queue them for display."""
        exchange_id = None
        for chunk in self.chat_obj.answer_question(self.prompt):
            self.question_answer_chunks_queue.put(chunk)
            exchange_id = chunk.exchange_id
        self.question_answer_chunks_queue.put(
            AssistantResponseChunk(exchange_id=exchange_id, content=None)
        )

    def play_queued_audios(self):
        """Play queued audio segments."""
        while True:
            try:
                logger.debug(
                    "Waiting for item from the audio reply chunk queue ({}) items so far",
                    self.chat_obj.play_speech_queue.qsize(),
                )
                speech_queue_item = self.chat_obj.play_speech_queue.get()
                audio = speech_queue_item["speech"]
                if audio is None:
                    logger.debug("Got `None`. No more audio reply chunks to play")
                    self.chat_obj.play_speech_queue.task_done()
                    break

                logger.debug("Playing audio reply chunk ({}s)", audio.duration_seconds)
                self.app_page.render_custom_audio_player(
                    audio,
                    parent_element=self.app_page.status_msg_container,
                    autoplay=True,
                    hidden=True,
                )
                logger.debug(
                    "Done playing audio reply chunk ({}s)", audio.duration_seconds
                )
                self.chat_obj.play_speech_queue.task_done()
            except Exception as error:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Error playing audio reply chunk ({}s)", audio.duration_seconds
                )
                logger.error(error)
                break
            finally:
                self.app_page.status_msg_container.empty()

    def stream_text_and_audio_reply(self):
        """Stream the text and audio reply to the display."""
        text_reply_container = st.empty()
        audio_reply_container = st.empty()

        chunk = AssistantResponseChunk(exchange_id=None, content="")
        full_response = ""
        text_reply_container.markdown("▌")
        self.app_page.status_msg_container.empty()
        while chunk.content is not None:
            logger.trace("Waiting for text or audio chunks...")
            # Render text
            with contextlib.suppress(queue.Empty):
                chunk = self.question_answer_chunks_queue.get_nowait()
                if chunk.content is not None:
                    full_response += chunk.content
                    text_reply_container.markdown(full_response + "▌")
                self.question_answer_chunks_queue.task_done()

        text_reply_container.caption(datetime.datetime.now().replace(microsecond=0))
        text_reply_container.markdown(full_response)

        logger.debug("Waiting for the audio reply to finish...")
        self.chat_obj.play_speech_queue.join()

        logger.debug("Getting path to full audio file for the reply...")
        history_entry_for_this_reply = (
            self.chat_obj.context_handler.database.retrieve_history(
                exchange_id=chunk.exchange_id
            )
        )
        full_audio_fpath = history_entry_for_this_reply["reply_audio_file_path"].iloc[0]
        if full_audio_fpath is None:
            logger.warning("Path to full audio file not available")
        else:
            logger.debug("Got path to full audio file: {}", full_audio_fpath)
            self.app_page.render_custom_audio_player(
                full_audio_fpath, parent_element=audio_reply_container, autoplay=False
            )

        return {"text": full_response, "audio": full_audio_fpath}


@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server as recommended by the streamlit-webrtc developers."""
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Cannot use their TURN servers. "
            "Falling back to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = TwilioClient(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers


def filter_page_info_from_queue(app_page: "AppPage", the_queue: queue.Queue):
    """Filter `app_page`'s data from `queue` inplace. Return queue of items in `app_page`.

    **Use with original_queue.mutex!!**

    Args:
        app_page: The page whose entries should be removed.
        the_queue: The queue to be filtered.

    Returns:
        queue.Queue: The queue with only the entries from `app_page`.

    Example:
    ```
    with the_queue.mutex:
        this_page_data = remove_page_info_from_queue(app_page, the_queue)
    ```
    """
    queue_with_only_entries_from_other_pages = queue.Queue()
    items_from_page_queue = queue.Queue()
    while the_queue.queue:
        original_queue_entry = the_queue.queue.popleft()
        if original_queue_entry["page"].page_id == app_page.page_id:
            items_from_page_queue.put(original_queue_entry)
        else:
            queue_with_only_entries_from_other_pages.put(original_queue_entry)

    the_queue.queue = queue_with_only_entries_from_other_pages.queue
    return items_from_page_queue


@st.cache_data
def get_avatar_images():
    """Return the avatar images for the assistant and the user."""
    avatar_files_dir = GeneralDefinitions.APP_DIR / "data"
    assistant_avatar_file_path = avatar_files_dir / "assistant_avatar.png"
    user_avatar_file_path = avatar_files_dir / "user_avatar.png"
    assistant_avatar_image = Image.open(assistant_avatar_file_path)
    user_avatar_image = Image.open(user_avatar_file_path)

    return {"assistant": assistant_avatar_image, "user": user_avatar_image}


@st.cache_data
def load_chime(chime_type: str) -> AudioSegment:
    """Load a chime sound from the data directory."""
    return AudioSegment.from_file(
        GeneralDefinitions.APP_DIR / "data" / f"{chime_type}.wav", format="wav"
    )
