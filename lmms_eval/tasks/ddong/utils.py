def ddong_doc_to_visaul(doc):
    return [doc["video"]]


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
