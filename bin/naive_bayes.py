import numpy as np

def train_model(relevant_sample, irrelevant_sample):
    sample = np.concatenate((relevant_sample, irrelevant_sample))
    words = list(set(sample))
    words = words[1:]
    d = len(words)

    mat = np.ones((2, d), dtype=np.float)

    for w in relevant_sample:
        if w in words:
            idx = words.index(w)
            mat[0, idx] = mat[0, idx] + 1

    for w in irrelevant_sample:
        if w in words:
           idx = words.index(w)
           mat[1, idx] = mat[1, idx] + 1

    row_sum = np.sum(mat, axis=1)
    y0 = row_sum[0]/np.sum(row_sum)
    y1 = row_sum[1]/np.sum(row_sum)
    y = [y0, y1]
    mat = mat / row_sum[:, np.newaxis]
    xy = mat[0:2, :]
    
    return xy, y, words


def score(xy, y, data):
    xy0 = 1.
    xy1 = 1.
    for d in data:
        xy0 = xy0 * xy[0, d]
        xy1 = xy1 * xy[1, d]

    if xy0*y[0] + xy1*y[1] == 0:
        return 0
    
    yx0 = xy0 *y[0] / (xy0*y[0] + xy1*y[1])
    
    return yx0
