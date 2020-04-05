from collections import defaultdict

def accuracy(model, sample_set):
    accuracy = 0
    for sample in sample_set:
        if model.classify(sample.features) == sample.label:
            accuracy += 1
    return accuracy

def histogram(values):
    result = defaultdict(int)
    for v in values:
        result[v] = result[v] + 1
    return result

def argmax(hist):
    max_value = 0
    max_key = None
    for (k, v) in hist.items():
        if v > max_value:
            max_key = k
            max_value = v
    return max_key

def majority_vote_count(labeled_samples):
    hist = histogram(sample.label for sample in labeled_samples)
    return max((v for v in hist.values()), default=0)

def majority_vote(labeled_samples):
    hist = histogram(sample.label for sample in labeled_samples)
    return argmax(hist)
