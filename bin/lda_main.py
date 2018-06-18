from lda import LDA
import  numpy as np
import glob

sample = []


def to_word_list(context):
    context = context.replace(',', '')
    context = context.replace('.', '')
    context = context.split('\n')
    context = [c.split(' ') for c in context]
    result = []
    for c in context:
        c = [w.lower() for w in c]
        if '' in c:
            c.remove('')
            if len(c) > 0:
                result.append(c)
        else:
            result.append(c)

    return result


def create_dict(path, stop_words=None):
    all_path = glob.glob(path)
    sample = []
    for p in all_path:
        with open(p, 'rb') as f:
            context = f.read()
            sample.extend(to_word_list(context))

    tmp = []
    [tmp.extend(s) for s in sample]
    words = list(set(tmp))

    if stop_words != None:
        for w in stop_words:
            if w in words:
                words.remove(w)

    word_dic = {}
    idx_dic = {}
    for idx, w in enumerate(words):
        word_dic[w] = idx
        idx_dic[idx] = w

    return word_dic, idx_dic


def load_data(path):
    all_path = glob.glob(path)
    sample = []
    for p in all_path:
        with open(p, 'rb') as f:
            context = f.read()
            sample.extend(to_word_list(context))
    return sample


def word_to_idx(sample, words_dict):
    sample = [[words_dict[w] for w in s if w in words_dict] for s in sample]
    return sample


def show(W, Z, idx_dict):
    for w, z in zip(W, Z):
        line = {}
        for word, topic in zip(w, z):
            line[idx_dict[word]] = topic
        print line


def normalize(X):
    return X / np.sum(X, axis=1)[:, np.newaxis]


def normal_distr_attr(X):
    m, k = np.shape(X)
    mean = np.sum(X ,axis=0) / m
    cov = np.cov(X.T) + np.eye(k) * 1e-6
    return mean, cov


with open('./stop_words_eng.txt') as f:
    stop_words = f.read()
    stop_words = stop_words.replace('\r', '')
    stop_words = stop_words.split("\n")

full_path = "../data/*/*.txt"
train_path = "../data/train/*.txt"
target_path = '../data/train/exam.txt'


train_re_path = '../data/train/relevant.txt'
train_ir_path = '../data/train/irrelevant.txt'
test2_ir_path = '../data/test2/irrelevant.txt'
test2_re_path = '../data/test2/relevant.txt'
test1_ir_path = '../data/test1/irrelevant.txt'
test1_re_path = '../data/test1/relevant.txt'


words_dict, idx_dict = create_dict(full_path, stop_words)

train_X = load_data(train_path)
train_X = word_to_idx(train_X, words_dict)

lda = LDA(5)

lda.fit(train_X, words_dict.items())

test1_re_X = load_data(test1_re_path)
test1_re_X = word_to_idx(test1_re_X, words_dict)
test1_ir_X = load_data(test1_ir_path)
test1_ir_X = word_to_idx(test1_ir_X, words_dict)

test2_re_X = load_data(test2_re_path)
test2_re_X = word_to_idx(test2_re_X, words_dict)
test2_ir_X = load_data(test2_ir_path)
test2_ir_X = word_to_idx(test2_ir_X, words_dict)

target_X = load_data(target_path)
target_X = word_to_idx(target_X, words_dict)

train_re_X = load_data(train_re_path)
train_re_X = word_to_idx(train_re_X, words_dict)
train_ir_X = load_data(train_ir_path)
train_ir_X = word_to_idx(train_ir_X, words_dict)


print normal_distr_attr(lda.transform(target_X))
print normal_distr_attr(lda.transform(train_re_X))
print normal_distr_attr(lda.transform(train_ir_X))
print normal_distr_attr(lda.transform(test1_re_X))
print normal_distr_attr(lda.transform(test1_ir_X))
print normal_distr_attr(lda.transform(test2_re_X))
print normal_distr_attr(lda.transform(test2_ir_X))

mean, cov = normal_distr_attr(lda.transform(target_X))

from scipy.stats import multivariate_normal
thres = np.sum(np.diag(cov)) / 5

print np.sum(multivariate_normal.pdf(lda.transform(train_re_X), mean, cov) > thres) / float(len(train_re_X))
print np.sum(multivariate_normal.pdf(lda.transform(train_ir_X), mean, cov) > thres) / float(len(train_ir_X))
print np.sum(multivariate_normal.pdf(lda.transform(test1_re_X), mean, cov) > thres) / float(len(test1_re_X))
print np.sum(multivariate_normal.pdf(lda.transform(test1_ir_X), mean, cov) > thres) / float(len(test1_ir_X))
print np.sum(multivariate_normal.pdf(lda.transform(test2_re_X), mean, cov) > thres) / float(len(test2_re_X))
print np.sum(multivariate_normal.pdf(lda.transform(test2_ir_X), mean, cov) > thres) / float(len(test2_ir_X))
