from PIL import Image
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor
import os


def _extract_frame(args):
    video_path, frame_idx = args
    return Image.fromarray(
        VideoReader(video_path, ctx=cpu(0)).get_batch([frame_idx]).asnumpy()[0]
    )


def ddong_doc_to_visaul(doc):
    video_path = doc["video"]
    frame_indices = list(range(20))

    # CPU 코어 개수에 따라 자동으로 스레드 수 결정 (I/O 바운드, 많은 스레드 가능)
    num_workers = min(os.cpu_count() or 4, 16)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(video_path, i) for i in frame_indices]
        frames = list(executor.map(_extract_frame, args_list))

    # None 값 필터링 (오류 발생한 프레임 제거)
    frames = [f for f in frames if f is not None]
    return frames


def ddong_doc_to_text(doc):
    return doc["question"]


def ddong_doc_to_target(doc):
    return doc["answer"]


def ddong_process_results(doc, results):
    preds = results[0].strip()

    if len(preds) > 0:
        pred = preds[-1]
    else:
        pred = ""

    return {"exact_match": pred == doc["answer"]}
