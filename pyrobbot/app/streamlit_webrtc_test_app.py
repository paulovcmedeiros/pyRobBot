# https://github.com/whitphx/streamlit-webrtc/blob/main/pages/10_sendonly_audio.py
from pyrobbot.app.app_page_templates import WebAppChat

# See
# <https://github.com/whitphx/streamlit-webrtc/blob/
# caf429e858fcf6eaf87096bfbb41be7e269cf0c0/streamlit_webrtc/mix.py#L33>
# and
# <https://github.com/whitphx/streamlit-webrtc/issues/361#issuecomment-894230158>
# for streamlit_webrtc hard-coded value for sample_rate = 48000


chat = WebAppChat()


def run_app():
    """Use WebRTC to transfer audio frames from the browser to the server."""


if __name__ == "__main__":
    run_app()
