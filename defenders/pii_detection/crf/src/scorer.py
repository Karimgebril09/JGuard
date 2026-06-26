def compute_local_score(X,feature_manager, weights,y, y_prev, i):
    """compute  local score for word in positionn i"""
    score = 0.0
    for feature in feature_manager.features:
        weight = weights.get(feature.feature_name, 0.0)
        score += weight * feature.compute(X,y_prev,y,i)
    return score



def precompute_scores(X, feature_manager, weights, labels):
    """precompute the scores for the incoming sequence of words X"""
    steps=len(X)
    scores = [{} for _ in range(steps)]
    prev_labels=["<START>"] + labels
    for t in range(steps):
        for label_prev in prev_labels:
            scores[t][label_prev] = {}
            for label_curr in labels:
                score = compute_local_score(X, feature_manager, weights, label_curr, label_prev, t)
                scores[t][label_prev][label_curr] = score # score when we do transistion from previous label to current label at position t

    return scores