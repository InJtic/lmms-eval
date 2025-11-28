from gemini_api.load import load_data
from dotenv import load_dotenv
import os
import requests
import json
from sys import stderr
from time import time


def save() -> tuple[float, float]:
    load_dotenv("../.env")
    api_key = os.environ["FACTCHAT_API_KEY"]

    start = time()
    with open(
        "../output/results/Gemini-2.5-flash/samples_ddong_bench.jsonl", "w"
    ) as output_f:
        for i, (content, raw) in enumerate(load_data("../ddong_eval_data.jsonl")):
            payload = {
                "model": "gemini-2.5-flash",
                "contents": [content],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 100,
                },
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            try:
                response = requests.post(
                    "https://factchat-cloud.mindlogic.ai/v1/api/google/models/generate-content",
                    headers=headers,
                    data=json.dumps(payload),
                )
                response.raise_for_status()
                result = response.json()

                try:
                    generated_text = result["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]
                except (KeyError, IndexError):
                    generated_text = ""
                    print(f"Warning: Empty response for doc_id {i}", file=stderr)

                text: str = generated_text.strip()

                matched = text == raw["answer"]

                output_entry = {
                    "doc_id": i,
                    "doc_hash": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
                    "input": raw["question"],
                    "target": raw["answer"],
                    "filtered_resps": [text],
                    "exact_match": matched,
                }

                output_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                output_f.flush()

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                continue
            except Exception as e:
                print(f"Error parsing response: {e}")
                continue

    end = time()
    return start, end
