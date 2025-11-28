from gemini_api.save import save
from gemini_api.generate_results import generate_results

if __name__ == "__main__":
    start, end = save()
    generate_results(start, end)
