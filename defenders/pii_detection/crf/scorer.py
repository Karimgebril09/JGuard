def compute_local_score(X,feature_manager, weights,y, y_prev, i):
    # compute weighed sum of feature at current position
    score = 0.0
    for feature in feature_manager.features:
        weight = weights.get(feature.feature_name, 0.0)
        score += weight * feature.compute(X,y_prev,y,i)
    return score



def precompute_scores(X, feature_manager, weights, labels):
    # precompute scores for all possible label transitions at each position
    steps=len(X)
    scores = [{} for _ in range(steps)]
    prev_labels=["<START>"] + labels
    for t in range(steps):
        for label_prev in prev_labels:
            scores[t][label_prev] = {}
            for label_curr in labels:
                score = compute_local_score(X, feature_manager, weights, label_curr, label_prev, t)
                scores[t][label_prev][label_curr] = score

    return scores