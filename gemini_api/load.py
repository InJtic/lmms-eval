import json
import cv2
from typing import Iterator, TypedDict
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
    print(file_path)
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            print(f"accessing {i}th line of {file_path}")
            if not line.strip():
                continue
            raw: Schema = json.loads(line)

            if raw["raw_label"] not in "01":
                continue

            cap = cv2.VideoCapture(raw["video"])

            image_parts: list[DataPart] = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                _, buffer = cv2.imencode(".png", frame)
                b64_img = base64.b64encode(buffer).decode("utf-8")

                image_parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": b64_img,
                        }
                    }
                )

            print("Got frames!")

            text_part: TextPart = {"text": raw["question"]}

            parts: list[TextPart | DataPart] = [text_part] + image_parts

            content: Content = {"role": "user", "parts": parts}

            print("Got Content!")

            yield content, raw
