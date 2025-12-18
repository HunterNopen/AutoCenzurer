from typing import TypedDict

class WordSchema(TypedDict):
    audio_id: str
    word_id: int
    word: str
    start_time: float
    end_time: float