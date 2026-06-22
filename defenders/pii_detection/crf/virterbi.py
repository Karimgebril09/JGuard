from defenders.pii_detection.crf.scorer import precompute_scores
START_TAG = "<START>"

def decode_using_viterbi(X, feature_manager, weights, labels):
    scores = precompute_scores(X,feature_manager, weights,labels)
    len_X = len(X)
    best_path = [{} for _ in range(len_X)]
    best_score = [{} for _ in range(len_X)]
    #initialize
    for label in labels:
        best_score[0][label] = scores[0][START_TAG][label] 
        best_path[0][label] = START_TAG


    
    for step in range(1, len_X):
        for curr_label in labels:
            best_score[step][curr_label] = float('-inf')

            for prev_label in labels:
                score = best_score[step-1][prev_label] + scores[step][prev_label][curr_label] # saved best score from previous step + current score
                if score > best_score[step][curr_label]:
                    best_score[step][curr_label] = score
                    best_path[step][curr_label] = prev_label

   
    #get the best final label
    best_final_label=None
    best_final_score=float('-inf')
    for label in labels:
        if best_score[len_X-1][label] > best_final_score:
            best_final_score = best_score[len_X-1][label]
            best_final_label = label

    # reconstruct reverse
    best_labeling=[best_final_label]
    while best_labeling[-1] != START_TAG:
        best_labeling.append(best_path[len_X-len(best_labeling)][best_labeling[-1]])

    best_labeling.reverse()
    return best_labeling[1:], best_final_score    