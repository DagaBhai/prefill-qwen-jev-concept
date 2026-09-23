import torch
from prefill_jev import FormatPrompt, tokenizer, model, get_token_id

def evaluate(state, text, choices):
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