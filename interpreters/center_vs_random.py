import pandas as pd
import json
import re

PATTERN = re.compile(r"(\d+)/(\d+)\.avi")


def interpret(
    samples_jsonl: str,
    data_jsonl: str = "data/center_vs_random/data.jsonl",
    metadata_csv: str = "data/center_vs_random/metadata.csv",
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

    center_2 = []
    center_4 = []
    random_2 = []
    random_4 = []

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
        position: str = info["position"]

        if position == "centered":
            if label in "01":
                center_2.append(result)
            else:
                center_4.append(result)
        else:
            if label in "01":
                random_2.append(result)
            else:
                random_4.append(result)

    print(
        f"Centered Mask with Binary Options: {sum(center_2) / len(center_2)}\n",
        f"Centered Mask with Quad Options: {sum(center_4) / len(center_4)}\n",
        f"Randomized Mask with Binary Options: {sum(random_2) / len(random_2)}\n",
        f"Randomized Mask with Quad Options: {sum(random_4) / len(random_4)}",
    )


if __name__ == "__main__":
    print("==========[Lexius__Phi-4-multimodal-instruct]==========")
    interpret(
        samples_jsonl="output/results/Lexius__Phi-4-multimodal-instruct/20251129_161831_samples_ddong_center_vs_random.jsonl"
    )
    print("\n==========[OpenGVLab__InternVL3_5-8B]==========")
    interpret(
        samples_jsonl="output/results/OpenGVLab__InternVL3_5-8B/20251129_162844_samples_ddong_center_vs_random.jsonl"
    )
    print("\n==========[Qwen__Qwen3-VL-8B-Instruct]==========")
    interpret(
        samples_jsonl="output/results/Qwen__Qwen3-VL-8B-Instruct/20251129_162038_samples_ddong_center_vs_random.jsonl"
    )
