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