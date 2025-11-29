from gemini_api.load import load_data
from dotenv import load_dotenv
import os
import json
import asyncio
import aiohttp
from sys import stderr
from time import time


# 재시도 로직을 포함한 비동기 요청 함수
async def request_with_retry(session, url, headers, payload, max_retries=5):
    retry_delay = 1  # 초기 대기 시간 (초)

    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 429:
                    print(
                        f"⚠️ 429 Rate Limit Hit. Waiting {retry_delay}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 대기 시간 2배 증가 (1, 2, 4, 8, 16...)
                    continue

                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

    return None


async def process_item(session, url, headers, content, raw, i, output_f):
    print(f"Processing step {i}...")

    payload = {
        "model": "gemini-2.5-flash",
        "contents": [content],
    }

    try:
        result = await request_with_retry(session, url, headers, payload)

        if result is None:
            print(f"Failed to get result for {i}")
            return

        try:
            generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            generated_text = ""
            print(f"Warning: Empty response for doc_id {i}", file=stderr)

        text = generated_text.strip()
        matched = text == raw["answer"]

        output_entry = {
            "doc_id": i,
            "doc_hash": "custom_hash",
            "input": raw["question"],
            "target": raw["answer"],
            "filtered_resps": [text],
            "exact_match": matched,
        }

        # 파일 쓰기는 동기 작업이므로 간단히 처리하거나 aiofiles를 쓸 수 있음.
        # 여기서는 간단히 일반 file write 사용 (버퍼링 주의)
        output_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
        output_f.flush()

    except Exception as e:
        print(f"Error processing item {i}: {e}")


async def save_async(path: str) -> tuple[float, float]:
    load_dotenv("./.env")
    api_key = os.environ["FACTCHAT_API_KEY"]
    url = "https://factchat-cloud.mindlogic.ai/v1/api/google/models/generate-content"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    start = time()
    print(f"Start Time: {start}", flush=True)

    # 출력 디렉토리 확인
    os.makedirs("output/results/Gemini-2.5-flash", exist_ok=True)

    async with aiohttp.ClientSession() as session:
        with open(
            "output/results/Gemini-2.5-flash/samples_ddong_bench.jsonl", "w"
        ) as output_f:
            # load_data는 generator이므로 여기서 순회
            # 동시성을 높이려면 Task를 모아서 gather를 써야 하지만,
            # 순서 보장 및 메모리 관리를 위해 하나씩 await로 처리 (속도는 requests보다 빠름)
            for i, (content, raw) in enumerate(load_data(path)):
                await process_item(session, url, headers, content, raw, i, output_f)

    end = time()
    return start, end


# 동기 코드와의 호환성을 위한 래퍼 함수
def save(path: str) -> tuple[float, float]:
    return asyncio.run(save_async(path))
