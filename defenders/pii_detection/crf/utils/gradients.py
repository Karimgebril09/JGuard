from math import exp

from defenders.pii_detection.crf.utils.forward_backward import forward_algorithm,backward_algorithm


def empirical_feature_counts(X,Y,feature_manager):
    """
    Compute empirical feature counts.

    count[k] = Σ f_k(y_{i-1}, y_i, x, i)
    """

    counts = {}

    for feature in feature_manager.features:
        counts[feature.feature_name] = 0.0

    prev = "<START>"

    for i in range(len(X)):

        curr = Y[i]

        for feature in feature_manager.features:

            counts[feature.feature_name] += feature.compute(
                X,
                prev,
                curr,
                i,
            )

        prev = curr

    return counts


def expected_feature_counts(X,feature_manager,weights,labels):
    """
    Compute expected feature counts under
    P(y|x).
    """

    alpha, logZ, scores = forward_algorithm(
        X,
        feature_manager,
        weights,
        labels,
    )

    beta = backward_algorithm(
        X,
        scores,
        labels,
    )

    expectations = {}

    for feature in feature_manager.features:
        expectations[feature.feature_name] = 0.0

    T = len(X)

    for t in range(T):

        if t == 0:

            previous_labels = ["<START>"]

        else:

            previous_labels = labels

        for prev in previous_labels:

            for curr in labels:

                if t == 0:

                    alpha_prev = 0.0

                else:

                    alpha_prev = alpha[t - 1][prev]

                probability = exp(
                    alpha_prev
                    + scores[t][prev][curr]
                    + beta[t][curr]
                    - logZ
                )

                for feature in feature_manager.features:

                    value = feature.compute(
                        X,
                        prev,
                        curr,
                        t,
                    )

                    expectations[
                        feature.feature_name
                    ] += probability * value

    return expectations


def compute_gradient(X,Y,feature_manager,weights,labels,l2=0.0):
    """
    Gradient of one training example.

    gradient =
    empirical
    -
    expected
    -
    λw
    """

    empirical = empirical_feature_counts(
        X,
        Y,
        feature_manager,
    )

    expected = expected_feature_counts(
        X,
        feature_manager,
        weights,
        labels,
    )

    gradient = {}

    for feature in feature_manager.features:

        name = feature.feature_name

        gradient[name] = (
            empirical[name]
            - expected[name]
            - l2 * weights[name]
        )

    return gradient