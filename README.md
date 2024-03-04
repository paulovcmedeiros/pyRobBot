<div align="center">

[![pyrobbot-logo](https://github.com/paulovcmedeiros/pyRobBot/blob/main/pyrobbot/app/data/assistant_avatar.png?raw=true)]((https://github.com/paulovcmedeiros/pyRobBot))
# <code>[pyRobBot](https://github.com/paulovcmedeiros/pyRobBot)</code><br>Chat with GPT LLMs over voice, text or both.<br>All with access to the internet.

[![Pepy Total Downlods](https://img.shields.io/pepy/dt/pyrobbot?style=flat&label=Downloads)](https://www.pepy.tech/projects/pyrobbot)
[![PyPI - Version](https://img.shields.io/pypi/v/pyrobbot)](https://pypi.org/project/pyrobbot/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pyrobbot.streamlit.app)
[<img src="https://raw.githubusercontent.com/paulovcmedeiros/pyRobBot/107f4576463d56b8d55bd913a56507940a37b675/pyrobbot/app/data/powered-by-openai-badge-outlined-on-dark.svg" width="100">](https://openai.com/blog/openai-api)


[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Contributors Welcome](https://img.shields.io/badge/Contributors-welcome-<COLOR>.svg)](https://github.com/paulovcmedeiros/pyRobBot/pulls)
[![Linting](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/linting.yaml/badge.svg)](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/linting.yaml)
[![Tests](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/tests.yaml/badge.svg)](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/paulovcmedeiros/pyRobBot/graph/badge.svg?token=XI8G1WH9O6)](https://codecov.io/gh/paulovcmedeiros/pyRobBot)

</div>

PyRobBot is a python package that uses OpenAI's [GPT large language models (LLMs)](https://platform.openai.com/docs/models) to implement a fully configurable **personal assistant** that, on top of the traditional chatbot interface, can also speak and listen to you using AI-generated **human-like** voices.


## Features

Features include, but are not limited to:

- [x] Voice Chat
  - Continuous voice input and output
  - No need to press a button: the assistant will keep listening until you stop talking

- [x] Internet access: The assistent will **search the web** to find the answers it doesn't have in its training data
  - E.g. latest news, current events, weather forecasts, etc.
  - Powered by [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)

- [x] Web browser user interface
    - See our [demo app on Streamlit Community Cloud](https://pyrobbot.streamlit.app)
  - Voice chat with:
    - **Continuous voice input and output**  (using [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc))
    - If you prefer, manual on/off toggling of the microphone (using [streamlit_mic_recorder](https://github.com/B4PT0R/streamlit-mic-recorder))
  - A familiar text interface integrated with the voice chat, for those who prefer a traditional chatbot experience
    - Your voice prompts and the assistant's voice replies are shown as text in the chat window
    - You may also send promts as text even when voice detection is enabled
  - Add/remove conversations dynamically
  - Automatic/editable conversation summary title
  - Autosave & retrieve chat history
    - Resume even the text & voice conversations started outside the web interface


- [x] Chat via terminal
  - For a more "Wake up, Neo" experience

- [x] Fully configurable
  - Large number of supported languages (*e.g.*, `rob --lang pt-br`)
  - Support for multiple LLMs through the OpenAI API
  - Choose your preferred Text-to-Speech (TTS) and Speech-To-Text (STT) engines (google/openai)
  - Control over the parameters passed to the OpenAI API, with (hopefully) sensible defaults
  - Ability to pass base directives to the LLM
    - E.g., to make it adopt a persona, but you decide which directived to pass
  - Dynamically modifiable AI parameters in each chat separately
    - No need to restart the chat

- [x] Chat context handling using [embeddings](https://platform.openai.com/docs/guides/embeddings)
- [x] Estimated API token usage and associated costs
- [x] OpenAI API key is **never** stored on disk



## System Requirements
- Python >= 3.9
- A valid [OpenAI API key](https://platform.openai.com/account/api-keys)
  - Set it in the Web UI or through the environment variable `OPENAI_API_KEY`
- To enable voice chat, you also need:
  - [PortAudio](https://www.portaudio.com/docs/v19-doxydocs/index.html)
    - Install on Ubuntu with `sudo apt-get --assume-yes install portaudio19-dev python-all-dev`
    - Install on CentOS/RHEL with `sudo yum install portaudio portaudio-devel`
  - [ffmpeg](https://ffmpeg.org/download.html)
    - Install on Ubuntu with `sudo apt-get --assume-yes install ffmpeg`
    - Install on CentOS/RHEL with `sudo yum install ffmpeg`

## Installation
This, naturally, assumes your system fulfills all [requirements](#system-requirements).

### Regular Installation
The recommended way for most users.

#### Using pip
```shell
pip install pyrobbot
```
#### From the GitHub repository
```shell
pip install git+https://github.com/paulovcmedeiros/pyRobBot.git
```

### Developer-Mode Installation
The recommended way for those who want to contribute to the project. We use [poetry](https://python-poetry.org) with the [poethepoet](https://poethepoet.natn.io/index.html) plugin. To get everything set up, run:
```shell
# Clean eventual previous install
curl -sSL https://install.python-poetry.org | python3 - --uninstall
rm -rf ${HOME}/.cache/pypoetry/ ${HOME}/.local/bin/poetry ${HOME}/.local/share/pypoetry
# Download and install poetry
curl -sSL https://install.python-poetry.org | python3 -
# Install needed poetry plugin(s)
poetry self add 'poethepoet[poetry_plugin]'
```


## Basic Usage
Upon succesfull installation, you should be able to run
```shell
rob [opts] SUBCOMMAND [subcommand_opts]
```
where `[opts]` and `[subcommand_opts]` denote optional command line arguments
that apply, respectively, to `rob` in general and to `SUBCOMMAND`
specifically.

**Please run `rob -h` for information** about the supported subcommands
and general `rob` options. For info about specific subcommands and the
options that apply to them only, **please run `rob SUBCOMMAND -h`** (note
that the `-h` goes after the subcommand in this case).

### Using the Web UI (defult, supports voice & text chat)
```shell
rob
```
See also our [demo Streamlit app](https://pyrobbot.streamlit.app)!

### Chatting Only by Voice
```shell
rob voice
```

### Running on the Terminal
```shell
rob .
```

## Disclaimers
This project's main purpose has been to serve as a learning exercise for me, as well as tool for experimenting with OpenAI API, GPT LLMs and text-to-speech/speech-to-text.

While it does not claim to be the best or more robust OpenAI-powered chatbot out there, it *does* aim to provide a friendly user interface that is easy to install, use and configure.

Feel free to open an [issue](https://github.com/paulovcmedeiros/pyRobBot/issues) or, even better, [submit a pull request](https://github.com/paulovcmedeiros/pyRobBot/pulls) if you find a bug or have a suggestion.

Last but not least: This project is **independently developed** and **not** affiliated, endorsed, or sponsored by OpenAI in any way. It is separate and distinct from OpenAI’s own products and services.
