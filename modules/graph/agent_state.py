from typing import TypedDict


class AgentState(TypedDict, total=False):
    input: str
    response: str = None
    last_node: str = None
