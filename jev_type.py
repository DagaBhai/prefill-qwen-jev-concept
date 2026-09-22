import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dic = {
    "state": "Customer wants to return a damaged product.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Choose the department that should handle this request.",
            "choices": ["returns", "shipping", "billing"]
        },
        "escalate": {
            "type": "noul",
            "instructions": "Does this message request a refund?"
        },
        "frustration": {
            "type": "score",
            "instructions": "Rate the customer's level of frustration.",
            "legend": {
                "0": "Calm",
                "1": "Frustrated",
                "2": "Very angry"
            }
        }
    }
}

model_name = "Qwen/Qwen3-1.7B"
cache_path = r"E:\abliteration\cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="cuda",
    cache_dir = cache_path
)

model.eval()

class FormatPrompt:
    @staticmethod
    def format(state, text, choices):
        options = "\n".join( f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))

        try:
          prompt = f"""STATE: {state}
          INSTRUCTION: {text}
          OPTIONS: {options}
          ANSWER:"""
          return prompt
        except:
          raise ValueError("Either question or instructions must be provided")

def get_token_id(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1: 
      raise ValueError( f"'{text}' must tokenize to exactly one token. Got {ids}" ) 
    return ids[0]


def run(state, text, choices):
  prompt = FormatPrompt.format(state, text, choices)

  prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

  with torch.no_grad():
      output = model(input_ids=prompt_ids, use_cache=True)

  logits = output.logits

  option_token_ids = torch.tensor( 
    [ get_token_id(chr(65 + i)) for i in range(len(choices)) ], 
    device=logits.device)
  
  option_logits = logits[:, -1, option_token_ids]
  probabilities = torch.softmax( option_logits.float(), dim=-1 )[0]
  return probabilities

def make_result(choices, probabilities): 
  probabilities = probabilities.detach().cpu()
  result = {choice: probability.item() 
            for choice, probability in zip( choices, probabilities )}
  sorted_probs = sorted(result.items(), key=lambda x: x[1], reverse=True)
  top = sorted_probs[0][1]
  second = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
  return { "choice": sorted_probs[0][0], "confidence": top - second, "probabilities": result }

output = {}

for key, value in dic["questions"].items():
    if value['type'] == 'choice':
        choices = value["choices"]
        probabilities = run(dic["state"], key, choices)
        result = make_result(choices, probabilities)
        output[key] = {"type": "choice", **result}

    if value['type'] == 'noul':
        choices = ["yes", "no"]
        probabilities = run(dic["state"], value["instructions"], choices)
        result = make_result( choices, probabilities )

        output[key] = {"type": "noul", "noul": result["probabilities"]["yes"], "probabilities": result["probabilities"] }

    if value['type'] == 'score':
        choices = list(value["legend"].keys())
        probabilities = run(dic["state"], value, choices)
        result = make_result( choices, probabilities )
        score = float(sum( int(choice) * probability for choice, probability in result["probabilities"].items()))
        output[key] = {"type": "score", "score": score, "confidence": result["confidence"], "probabilities": result["probabilities"]}


print(output)

"""
{
  "state": "Customer wants to return a damaged product.",
  "questions": {
    "department": {
      "type": "choice",
      "choices": ["returns", "shipping", "billing"]
    },
    "escalate": {
      "type": "noul"
    },
    "frustration": {
      "type": "score",
      "legend": {
        "0": "Calm",
        "1": "Frustrated",
        "2": "Very angry"
      }
    }
  }
}


{
  "department": {
    "type": "choice",
    "choice": "returns",
    "confidence": 0.21,
    "probabilities": {
      "returns": 0.47,
      "shipping": 0.28,
      "billing": 0.25
    }
  },
  "escalate": {
    "type": "noul",
    "noul": 0.93
  },
  "frustration": {
    "type": "score",
    "score": 1.44,
    "confidence": 0.78,
    "probabilities": {
      "0": 0.00,
      "1": 0.56,
      "2": 0.44
    }
  }
}
"""