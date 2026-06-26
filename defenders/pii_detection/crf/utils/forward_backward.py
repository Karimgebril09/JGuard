from defenders.pii_detection.crf.src.scorer import precompute_scores
import math

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def log_sum_exp(values):

    if len(values) == 0:
        return -float("inf")

    m = max(values)

    if m == -float("inf"):
        return m

    return m + math.log(sum(math.exp(v - m) for v in values))


def forward_algorithm(X,feature_manager,weights,labels):
    """
    Compute forward variables .

    Returns
    -------
    alpha
    logZ
    scores
    """

    scores = precompute_scores(
        X,
        feature_manager,
        weights,
        labels,
    )

    T = len(X)

    alpha = []

    first = {}

    for curr in labels:

        first[curr] = scores[0][START_TAG][curr]

    alpha.append(first)

    for t in range(1, T):

        current = {}

        for curr in labels:

            values = []

            for prev in labels:

                values.append(
                    alpha[t - 1][prev]
                    + scores[t][prev][curr]
                )

            current[curr] = log_sum_exp(values)

        alpha.append(current)

    logZ = log_sum_exp(
        alpha[T - 1].values()
    )

    return alpha, logZ, scores


def backward_algorithm(X,scores,labels):
    """
    Compute backward variables β.
    """

    T = len(X)

    beta = [{} for _ in range(T)]

    for y in labels:
        beta[T - 1][y] = 0.0

    for t in range(T - 2, -1, -1):

        for prev in labels:

            values = []

            for curr in labels:

                values.append(
                    scores[t + 1][prev][curr]
                    + beta[t + 1][curr]
                )

            beta[t][prev] = log_sum_exp(values)

    return beta