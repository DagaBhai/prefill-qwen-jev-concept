from prefill_jev import make_result, evaluate

def system_one(state, questions: dict):
    output = {}
    for key, q in questions.items():
        qtype = q["type"]

        if qtype == "choice":
            choices = list(q["criteria"].keys())
            probabilities = evaluate(state, q["instructions"], choices)
            result = make_result(choices, probabilities)
            output[key] = {"type": "choice", **result}

        elif qtype == "noul":
            choices = ["yes", "no"]
            probabilities = evaluate(state, q["instructions"], choices)
            result = make_result(choices, probabilities)
            output[key] = {"type": "noul", "noul": result["probabilities"]["yes"], "probabilities": result["probabilities"]}

        elif qtype == "score":
            legend = {str(i): label for i, label in enumerate(q["criteria"])}
            choices = list(legend.keys())
            probabilities = evaluate(state, q["instructions"], choices)
            result = make_result(choices, probabilities)
            score = float(sum(int(c) * p for c, p in result["probabilities"].items()))
            output[key] = {"type": "score", "score": score, "legend": legend, "confidence": result["confidence"], "probabilities": result["probabilities"]}

    return output