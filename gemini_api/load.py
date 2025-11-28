import json
from typing import Iterator, TypedDict, cast
import mimetypes
import base64


class Schema(TypedDict):
    video: str
    question: str
    answer: str
    raw_label: str


class TextPart(TypedDict):
    text: str


class Blob(TypedDict):
    mimeType: str
    data: str


class DataPart(TypedDict):
    inlineData: Blob


class Content(TypedDict):
    role: str
    parts: list[TextPart | DataPart]


def load_data(file_path: str) -> Iterator[tuple[Content, Schema]]:
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            raw: Schema = json.loads(line)

            if raw["answer"] not in "01":
                continue

            mime_type, _ = mimetypes.guess_type(raw["video"])

            mime_type = cast(str, mime_type)

            with open(raw["video"], "rb") as f:
                video_b64_str = base64.b64encode(f.read()).decode("utf-8")

            content: Content = {
                "role": "user",
                "parts": [
                    {
                        "text": raw["question"],
                    },
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": video_b64_str,
                        }
                    },
                ],
            }

            yield content, raw
