import torch
import numpy as np

def get_answer(model, tokenizer, prompt):
    """Generate answer using Qwen."""
    try:
        torch.cuda.empty_cache()
        choices = ['A', 'B', 'C', 'D']
        choice_ids = [tokenizer.encode(choice)[-1] for choice in choices]
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            last_token_logits = outputs.logits[:, -1, :]
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            row_softmax = list(softmax(choice_logits[0]))
            pred = choices[np.argmax(row_softmax)]
            row_softmax = [float(s) for s in row_softmax]
        return pred, row_softmax

    except Exception as e:
        print(f"Error during generation: {e}")
        torch.cuda.empty_cache()
        raise e

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def ppl(text, model, tokenizer):
    """Source: https://huggingface.co/docs/transformers/perplexity"""
    encodings = tokenizer(text, return_tensors="pt").to(model.device)
    max_length = tokenizer.model_max_length
    stride = max_length
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    perplexity = torch.exp(avg_nll).cpu().item()
    return perplexity