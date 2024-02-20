<div align="center">

[![pyrobbot-logo](./pyrobbot/app/data/assistant_avatar.png)]((https://github.com/paulovcmedeiros/pyRobBot))
# <code>[pyRobBot](https://github.com/paulovcmedeiros/pyRobBot)</code><br>Chat with GPT LLMs over voice, UI & terminal.<br>All with access to the internet.

[![Pepy Total Downlods](https://img.shields.io/pepy/dt/pyrobbot?style=flat&label=Downloads)](https://www.pepy.tech/projects/pyrobbot)
[![PyPI - Version](https://img.shields.io/pypi/v/pyrobbot)](https://pypi.org/project/pyrobbot/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pyrobbot.streamlit.app)
[<img src="./pyrobbot/app/data/powered-by-openai-badge-outlined-on-dark.svg" width="100">](https://openai.com/blog/openai-api)


[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Contributors Welcome](https://img.shields.io/badge/Contributors-welcome-<COLOR>.svg)](https://github.com/paulovcmedeiros/pyRobBot/pulls)
[![Linting](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/linting.yaml/badge.svg)](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/linting.yaml)
[![Tests](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/tests.yaml/badge.svg)](https://github.com/paulovcmedeiros/pyRobBot/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/paulovcmedeiros/pyRobBot/graph/badge.svg?token=XI8G1WH9O6)](https://codecov.io/gh/paulovcmedeiros/pyRobBot)

</div>

PyRobBot is a python package that uses OpenAI's [GPT large language models (LLMs)](https://platform.openai.com/docs/models) to implement:
* A fully configurable **personal assistant** that can speak and listen to you
* An equally fully configurable text-based **chatbot** that can be used either via web UI or terminal


## Features
- [x] Personal assistant with text-to-speech and speech-to-text capabilities
  - Talk to the GPT assistant and the assistant will talk back to you!
  - Choose your preferred language (e.g., `rob --lang pt-br`)
  - Choose your preferred Text-to-Speech (TTS) engine (e.g., `rob --tts google`)
    - [OpenAI Text-to-Speech](https://platform.openai.com/docs/guides/text-to-speech) (default): AI-generated *human-like* voice
    - [Google TTS](https://cloud.google.com/text-to-speech): free at the time being, with decent quality
  - Choose your preferred Speech-to-Text (STT) engine
    - Also between OpenAI and Google (default)
- [x] Web browser UI (made with [Streamlit](https://pyrobbot.streamlit.app))
  - Add/remove conversations dynamically
  - Automatic/editable conversation summary title
  - Input via text or voice
- [x] Terminal UI
  - For a more "Wake up, Neo" experience
- [x] Internet access: The assistent will **search the web** and to find the answers it doesn't have in its training data
  - E.g. current events, weather forecasts, etc.
- [x] Fully configurable
  - Support for multiple GPT LLMs
  - Control over the parameters passed to the OpenAI API, with (hopefully) sensible defaults
  - Ability to pass base directives to the LLM
    - E.g., to make it adopt a persona, but you decide which directived to pass
  - Dynamically modifiable AI parameters in each chat separately
    - No need to restart the chat
- [x] Autosave & retrieve chat history
  - In the browser UI, you can even read the transcripts of your voice conversations with the AI
- [x] Chat context handling using [embeddings](https://platform.openai.com/docs/guides/embeddings)
- [x] Estimated API token usage and associated costs
- [x] OpenAI API key is **never** stored on disk



## System Requirements
- Python >= 3.9
- A valid [OpenAI API key](https://platform.openai.com/account/api-keys)
  - Set in the Web UI or through the environment variable `OPENAI_API_KEY`
- To enable voice chat, you also need:
  - [PortAudio](https://www.portaudio.com/docs/v19-doxydocs/index.html)
    - Install on Ubuntu with `sudo apt-get --assume-yes install portaudio19-dev python-all-dev`
    - Install on CentOS/RHEL with `sudo yum install portaudio portaudio-devel`
  - [ffmpeg](https://ffmpeg.org/download.html)
    - Install on Ubuntu with `sudo apt-get --assume-yes install ffmpeg`
    - Install on CentOS/RHEL with `sudo yum install ffmpeg`

## Installation
This, naturally, assumes your system fulfills all [requirements](#system-requirements).
### Using pip
```shell
pip install pyrobbot
```

### From source
```shell
pip install git+https://github.com/paulovcmedeiros/pyRobBot.git
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

### Chatting by Voice (default)
```shell
rob
```

### Using the Web UI
```shell
rob ui
```
See also our [demo Streamlit app](https://pyrobbot.streamlit.app)!

### Running on the Terminal
```shell
rob .
```

## Disclaimers
This project's main purpose has been to serve as a learning exercise for me, as well as tool for experimenting with OpenAI API, GPT LLMs and text-to-speech/speech-to-text.

While it does not claim to be the best or more robust OpenAI-powered chatbot out there, it *does* aim to provide a friendly user interface that is easy to install, use and configure.

Feel free to open an [issue](https://github.com/paulovcmedeiros/pyRobBot/issues) or, even better, [submit a pull request](https://github.com/paulovcmedeiros/pyRobBot/pulls) if you find a bug or have a suggestion.

Last but not least: This project is **independently developed** and **not** affiliated, endorsed, or sponsored by OpenAI in any way. It is separate and distinct from OpenAI’s own products and services.
