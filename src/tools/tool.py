from abc import ABC, abstractmethod


class MetaTool(ABC):

    def __init__(self) -> None:
        self.tool_name: str = None  # e.g., pytorch
        self.tool_tutorial: str = None  # e.g., readme.md
        self.tool_description: str = None  # e.g., usage introduction (summary)

    def calling(self, goal: str, function: dict, parameters: dict):  # e.g., get started
        pass


class GitHubTool(MetaTool):

    def __init__(self) -> None:
        super().__init__()
        self.url: str = None


class APITool(MetaTool):
    def __init__(self) -> None:
        super().__init__()


class OSTool(MetaTool):

    def __init__(self) -> None:
        super().__init__()


class ROSTool(OSTool):

    def __init__(self) -> None:
        super().__init__()


class MacOSTool(OSTool):

    def __init__(self) -> None:
        super().__init__()


class AndroidOSTool(OSTool):

    def __init__(self) -> None:
        super().__init__()


class AgentTool(MetaTool):

    def __init__(self) -> None:
        super().__init__()
        self.profile: str = None



