"""General utility functions and classes."""
import json
from pathlib import Path

from loguru import logger


class AlternativeConstructors:
    """Mixin class for alternative constructors."""

    @classmethod
    def from_dict(cls, configs: dict):
        """Creates an instance from a configuration dictionary.

        Converts the configuration dictionary into a instance of this class
        and uses it to instantiate the Chat class.

        Args:
            configs (dict): The configuration options as a dictionary.

        Returns:
            cls: An instance of Chat initialized with the given configurations.
        """
        return cls(configs=cls.default_configs.model_validate(configs))

    @classmethod
    def from_cli_args(cls, cli_args):
        """Creates an instance from CLI arguments.

        Extracts relevant options from the CLI arguments and initializes a class instance
        with them.

        Args:
            cli_args: The command line arguments.

        Returns:
            cls: An instance of the class initialized with CLI-specified configurations.
        """
        chat_opts = {
            k: v
            for k, v in vars(cli_args).items()
            if k in cls.default_configs.model_fields and v is not None
        }
        return cls.from_dict(chat_opts)

    @classmethod
    def from_cache(cls, cache_dir: Path):
        """Loads an instance from a cache directory.

        Args:
            cache_dir (Path): The path to the cache directory.

        Returns:
            cls: An instance of the class loaded with cached configurations and metadata.
        """
        try:
            with open(cache_dir / "configs.json", "r") as configs_f:
                new = cls.from_dict(json.load(configs_f))
            with open(cache_dir / "metadata.json", "r") as metadata_f:
                new.metadata = json.load(metadata_f)
                new.id = new.metadata["chat_id"]
        except FileNotFoundError:
            logger.warning(
                "Could not find configs and/or metadata file in cache directory. "
                + f"Creating {cls.__name__} with default configs."
            )
            new = cls()
        return new
