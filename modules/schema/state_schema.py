from typing import Optional
from pydantic import BaseModel


class StateSchema(BaseModel):
    input: str
    response: str
    next_node: Optional[str] = None
    last_node: Optional[str] = None
