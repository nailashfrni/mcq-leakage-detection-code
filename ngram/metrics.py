import editdistance

def exact_match_score(predicted_text, original_text):
    return int(predicted_text == original_text)

def edit_similarity_score(predicted_text, original_text):
    edit_dist = editdistance.eval(predicted_text, original_text)
    max_length = max(len(predicted_text), len(original_text))
    if max_length == 0:
      edit_similarity = 0
    else:
      edit_similarity = 1 - (edit_dist / max_length)
    return edit_similarity

def rouge_l_score(predicted_text, original_text, scorer):
    # Calculate Rouge-L score
    rouge_score = scorer.score(original_text, predicted_text)['rougeL'].fmeasure
    return rouge_score