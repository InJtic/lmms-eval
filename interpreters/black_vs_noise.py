import pandas as pd
import json
import re

PATTERN = re.compile(r"(\d+)/(\d+)\.avi")
NOISE_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def interpret(
    samples_jsonl: str,
    data_jsonl: str = "data/black_vs_noise/data.jsonl",
    metadata_csv: str = "data/black_vs_noise/metadata.csv",
):
    metadata = pd.read_csv(metadata_csv, index_col="seed")
    data_list = []
    samples = []

    with open(data_jsonl, "r") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    with open(samples_jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    black_2 = {_noise: [] for _noise in (1, 0.999, 0.99, 0.9, 0.8)}
    black_4 = {_noise: [] for _noise in (1, 0.999, 0.99, 0.9, 0.8)}
    noise_2 = {_noise: [] for _noise in (1, 0.999, 0.99, 0.9, 0.8)}
    noise_4 = {_noise: [] for _noise in (1, 0.999, 0.99, 0.9, 0.8)}

    for sample in samples:
        row = sample["doc_id"]
        result = 1 if sample["target"] == sample["filtered_resps"][0] else 0

        data = data_list[row]

        if (match := PATTERN.search(data["video"])) is not None:
            data_index = int(match.group(1))
            # sample_index = int(match.group(2))

        else:
            print(data["video"])
            raise ValueError

        info = metadata.loc[data_index]

        label: str = info["label"]
        fill: bool = info["fill"]
        noise: float = float(re.findall(NOISE_PATTERN, info["noise"])[0])

        if fill:
            if label in "01":
                black_2[noise].append(result)
            else:
                black_4[noise].append(result)
        else:
            if label in "01":
                noise_2[noise].append(result)
            else:
                noise_4[noise].append(result)

    for noise in (1, 0.999, 0.99, 0.9, 0.8):
        print(
            f"At noise {noise}, (Case: {len(black_4[noise])}, {len(black_2[noise])})\n"
            f"Black Mask with Binary Options: {sum(black_2[noise]) / len(black_2[noise])}\n",
            f"Black Mask with Quad Options: {sum(black_4[noise]) / len(black_4[noise])}\n",
            f"Noise Mask with Binary Options: {sum(noise_2[noise]) / len(noise_2[noise])}\n",
            f"Noise Mask with Quad Options: {sum(noise_4[noise]) / len(noise_4[noise])}",
        )


if __name__ == "__main__":
    print("==========[Lexius__Phi-4-multimodal-instruct]==========")
    interpret(
        samples_jsonl="output/results/Lexius__Phi-4-multimodal-instruct/20251129_161831_samples_ddong_black_vs_noise.jsonl"
    )
    print("\n==========[OpenGVLab__InternVL3_5-8B]==========")
    interpret(
        samples_jsonl="output/results/OpenGVLab__InternVL3_5-8B/20251129_162844_samples_ddong_black_vs_noise.jsonl"
    )
    print("\n==========[Qwen__Qwen3-VL-8B-Instruct]==========")
    interpret(
        samples_jsonl="output/results/Qwen__Qwen3-VL-8B-Instruct/20251129_162038_samples_ddong_black_vs_noise.jsonl"
    )
