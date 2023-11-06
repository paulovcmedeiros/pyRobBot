import pytest


@pytest.mark.parametrize("user_input", ("Hi!", ""), ids=("regular-input", "empty-input"))
def test_terminal_chat(default_chat):
    default_chat.start()
