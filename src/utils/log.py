from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Sequence

from loguru import logger


if TYPE_CHECKING:
    from loguru import Logger


class LogColors:
    """
    ANSI color codes for use in console output.
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BLACK = "\033[30m"

    BOLD = "\033[1m"
    ITALIC = "\033[3m"

    END = "\033[0m"

    @classmethod
    def get_all_colors(cls: type[LogColors]) -> list:
        names = dir(cls)
        names = [name for name in names if not name.startswith("__") and not callable(getattr(cls, name))]
        return [getattr(cls, name) for name in names]

    def render(self, text: str, color: str = "", style: str = "") -> str:
        """
        render text by input color and style.
        """
        colors = self.get_all_colors()
        if color and color in colors:
            error_message = f"color should be in: {colors} but now is: {color}"
            raise ValueError(error_message)
        if style and style in colors:
            error_message = f"style should be in: {colors} but now is: {style}"
            raise ValueError(error_message)

        text = f"{color}{text}{self.END}"

        return f"{style}{text}{self.END}"


class Logger:

    def __init__(self) -> None:
        self.logger: Logger = logger

    def info(self, *args: Sequence, plain: bool = False, title: str = "Info") -> None:
        if plain:
            return self.plain_info(*args)
        for arg in args:
            info = f"{LogColors.WHITE}{arg}{LogColors.END}"
            self.logger.info(info)
        return None

    def __getstate__(self) -> dict:
        return {}
    def __setstate__(self, _: str) -> None:
        self.logger = logger

    def plain_info(self, *args: Sequence) -> None:
        for arg in args:
            info = f"""
                {LogColors.YELLOW}{LogColors.BOLD}
                Info:{LogColors.END}{LogColors.WHITE}{arg}{LogColors.END}
            """
            self.logger.info(info)

    def warning(self, *args: Sequence) -> None:
        for arg in args:
            info = f"{LogColors.BLUE}{LogColors.BOLD}Warning:{LogColors.END}{arg}"
            self.logger.warning(info)

    def error(self, *args: Sequence) -> None:
        for arg in args:
            info = f"{LogColors.RED}{LogColors.BOLD}Error:{LogColors.END}{arg}"
            self.logger.error(info)

    def log_message(self, messages: list[dict[str, str]]) -> None:
        """
        messages is some info like this  [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        """
        with formatting_log(self.logger, "GPT Messages"):
            for m in messages:
                info = f"""
                    {LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END}
                    {LogColors.CYAN}{m['role']}{LogColors.END}\n
                    {LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END}
                    {LogColors.CYAN}{m['content']}{LogColors.END}\n
                """
                self.logger.info(info)

    def log_response(self, response: str) -> None:
        with formatting_log(self.logger, "GPT Response"):
            info = f"{LogColors.CYAN}{response}{LogColors.END}\n"
            self.logger.info(info)

