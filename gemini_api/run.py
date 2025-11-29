from gemini_api.save import save
from gemini_api.generate_results import generate_results

if __name__ == "__main__":
    for task in (
        "black_vs_noise",
        "center_vs_random",
        "color_vs_wb",
        "direction",
    ):
        print(task)
        start, end = save(f"data/{task}/data_light.jsonl")
        generate_results(start, end, task)
