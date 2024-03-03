import io
import subprocess

import pytest
from pydub import AudioSegment

from pyrobbot.__main__ import main
from pyrobbot.argparse_wrapper import get_parsed_args


def test_default_command():
    args = get_parsed_args(argv=[])
    assert args.command == "ui"


@pytest.mark.usefixtures("_input_builtin_mocker")
@pytest.mark.parametrize("user_input", ["Hi!", ""], ids=["regular-input", "empty-input"])
def test_terminal_command(cli_args_overrides):
    args = ["terminal", "--report-accounting-when-done", *cli_args_overrides]
    args = list(dict.fromkeys(args))
    main(args)


def test_accounting_command():
    main(["accounting"])


def test_ui_command(mocker, caplog):
    original_run = subprocess.run

    def new_run(*args, **kwargs):
        kwargs.pop("timeout", None)
        try:
            original_run(*args, **kwargs, timeout=0.5)
        except subprocess.TimeoutExpired as error:
            raise KeyboardInterrupt from error

    mocker.patch("subprocess.run", new=new_run)
    main(["ui"])
    assert "Exiting." in caplog.text


@pytest.mark.parametrize("stt", ["google", "openai"])
@pytest.mark.parametrize("tts", ["google", "openai"])
def test_voice_chat(mocker, mock_wav_bytes_string, tts, stt):
    # We allow even number of calls in order to let the function be tested first and
    # then terminate the chat
    def _mock_listen(*args, **kwargs):  # noqa: ARG001
        try:
            _mock_listen.execution_counter += 1
        except AttributeError:
            _mock_listen.execution_counter = 0
        if _mock_listen.execution_counter % 2:
            return None
        return AudioSegment.from_wav(io.BytesIO(mock_wav_bytes_string))

    mocker.patch("pyrobbot.voice_chat.VoiceChat.listen", _mock_listen)
    main(["voice", "--tts", tts, "--stt", stt])
