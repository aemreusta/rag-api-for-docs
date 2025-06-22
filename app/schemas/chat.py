from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    chat_history: list[Message] | None = None


class ChatResponse(BaseModel):
    message: str
