import numpy as np
import random
import glob
import os
from scipy.special import digamma, gammaln


class LDA():

    def __init__(self):
        pass
   
    def random_topic_assign(self, X, d):
        z = []
        for x in X:
            z.append(np.array(np.round(np.random.rand(len(x),)*(d-1)), dtype=np.int))
        return z

  
    def fit(self, X, dic, k):
        m = len(X)
        d = len(dic)
        self.m = len(X)
        self.d = len(dic)
        self.k = k
        self.dic = dic 
        self.W = X
        self.Z = self.random_topic_assign(X, k)
        self.nmk = np.zeros((m, k), np.int)
        self.ndk = np.zeros((d, k), np.int)
        
        for i in range(self.m):
            for j in range(len(self.W[i])):
                t = self.Z[i][j]
                w = self.W[i][j]
                self.nmk[i][t]+=1
                self.ndk[w][t]+=1

        self.nmksum = np.sum(self.nmk, axis=1)
        self.ndksum = np.sum(self.ndk, axis=0)
        self.alpha = 1. 
        self.beta = 1.
        self.eta = 1.

        self.learning_decay = .7
        self.learning_offset = 10.

    def dirichlet_expectation_2d(self, arr):
        n_rows = arr.shape[0]
        n_cols = arr.shape[1]
       
        d_exp = np.empty_like(arr)
        for i in range(n_rows):
            row_total = 0
            for j in range(n_cols):
                row_total += arr[i, j]
            psi_row_total = digamma(row_total)

            for j in range(n_cols):
                d_exp[i, j] = digamma(arr[i, j]) - psi_row_total

        return d_exp

    def mean_change(self, arr_1, arr_2):
        size = arr_1.shape[0]
        total = 0.0
        for i in range(size):
            diff = np.fabs(arr_1[i] - arr_2[i])
            total += diff
        return total / size

    def e_step(self, X, lamb):
        #gamma = np.random.gamma(100., 1./100., (self.m, self.k)) 
        gamma = np.ones((self.m, self.k))
        Elogbeta = self.dirichlet_expectation_2d(lamb)
        sstats = np.zeros((self.k, self.d))
       
        for i in range(100):
            last_gamma = gamma.copy()
            Elogtheta = self.dirichlet_expectation_2d(gamma)
           
            normphi = np.dot(np.exp(Elogtheta), np.exp(Elogbeta))

            for m, words in enumerate(self.W):
                phi = np.exp(Elogbeta[:, words] + Elogtheta[m,:, np.newaxis])
                phi = phi / normphi[m, words] 
                gamma[m] = self.alpha + np.sum(phi, axis=1)
            
            if np.sum(self.mean_change(last_gamma, gamma)) / self.k < 1e-3:
                break  
        
        for m, words in enumerate(self.W):
            phi = np.exp(Elogbeta[:, words] + Elogtheta[m,:, np.newaxis])
            phi = phi / normphi[m, words]
            for i, w in enumerate(words):
                sstats[:, w] += phi[:,i]

        return gamma, sstats

    def m_step(self, sstats, lamb, n_iter):
        rhot = np.power(self.learning_offset + n_iter,
                              -self.learning_decay)
        # return lamb * (1-rhot) + rhot * (self.eta + self.m * sstats / self.d)
        return self.eta + sstats

    def approx_bound(self, X, gamma, lamb):
        M = len(X)

        score = 0
        Elogtheta = self.dirichlet_expectation_2d(gamma)
        Elogbeta = self.dirichlet_expectation_2d(lamb)

        # E[log p(docs | theta, beta)]
        for d in range(0, M):
            gammad = gamma[d, :]
            x = X[d]
            phinorm = np.zeros(len(x))
            for i in range(0, len(x)):
                phi = Elogtheta[d, :] + Elogbeta[:, x[i]]
                phinorm[i] = np.log(sum(np.exp(phi)))
            score += np.sum(phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self.alpha - gamma)*Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self.alpha))
        score += np.sum(gammaln(self.alpha*self.k) - gammaln(np.sum(gamma, 1)))


        # E[log p(beta | eta) - log q (beta | lambda)]
        score += np.sum((self.eta - lamb)* Elogbeta)
        score += np.sum(gammaln(lamb) - gammaln(self.eta))
        score += np.sum(gammaln(self.eta*self.d) - gammaln(np.sum(lamb, 1)))

        return score 


    def em(self):
        lamb = np.random.gamma(100., 1./100., (self.k, self.d)) 
        #lamb = 1./self.d + np.random.rand(self.k, self.d)

        for n_iter in range(10):

            gamma, sstats = self.e_step(self.W, lamb)
            lamb = self.m_step(sstats, lamb, n_iter)
            bound = self.approx_bound(self.W, gamma, lamb)
            print bound
        
        return lamb 

    def mcmc_train(self):
        for n_iter in range(100):
            for m in range(self.m):
                for n in range(len(self.W[m])):
                    w = self.W[m][n]
                    t = self.Z[m][n]
                    self.nmk[m][t]-=1
                    self.ndk[w][t]-=1
                    self.ndksum[t]-=1
                    
                    tmp = self.beta*np.ones(self.k,)*self.d
                    p = (self.nmk[m] + self.alpha)*(self.ndk[w] + self.beta)/ (self.ndksum+tmp)
                    
                    for i in range(self.k-1):
                        p[i + 1] += p[i]

                    u = random.uniform(0, p[self.k-1])
                    for new_topic in range(self.k):
                        if p[new_topic] > u:
                            break
                    
                    self.Z[m][n] = new_topic
                    self.nmk[m][new_topic]+=1
                    self.ndk[w][new_topic]+=1
                    self.ndksum[new_topic]+=1
            print "iteration {}".format(n_iter)
            print "score {}".format(self.score())

    def score(self):
        p = 0.
        for i in range(self.m):
            for j in range(len(self.W[i])):
                w = self.W[i][j]
                z = self.Z[i][j]
                p += np.log((float(self.nmk[i][z]) / self.nmksum[i]) * (float(self.ndk[w][z]) / self.ndksum[z]))
        return p

    def save_model(self, path):
        with open(os.path.join(path,"Z.csv"), 'wb') as f:
            np.savetxt(f, self.Z, delimiter=",", fmt='%5s')
        with open(os.path.join(path,"W.csv"), 'wb') as f:
            np.savetxt(f, self.W, delimiter=",", fmt='%5s')

        with open(os.path.join(path,"nmk.csv"), 'wb') as f:
            np.savetxt(f, self.nmk, delimiter=",", fmt='%5s')
        with open(os.path.join(path,"ndk.csv"), 'wb') as f:
            np.savetxt(f, self.ndk, delimiter=",", fmt='%5s')

sample = []

def to_word_list(context): 
    context = context.replace(',','')
    context = context.replace('.','')
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
        idx_dic[idx] =  w        
    
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
    for w,z in zip(W,Z):
        line = {}
        for word, topic in zip(w,z):
            line[idx_dict[word]] = topic
        print line 


with open('./bin/stop_words_eng.txt') as f:
    stop_words = f.read()
    stop_words = stop_words.replace('\r','')
    stop_words =stop_words.split("\n")


full_path = "./data/*/*.txt"
train_path = "./data/train/*.txt"


words_dict, idx_dict = create_dict(full_path, stop_words)
sample = load_data(train_path)
sample = word_to_idx(sample, words_dict)

lda = LDA()
lda.fit(sample, words_dict.items(), 5)
lamb = lda.em()
print np.sum(lamb, axis=1)
print lamb
'''
lda.mcmc_train()
lda.save_model('./bin/')

print lda.ndksum
print lda.Z[0:10]
print lda.Z[-10:]
print show(lda.W[0:10], lda.Z[0:10], idx_dict)
print show(lda.W[-10:], lda.Z[-10:], idx_dict)
'''

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5, verbose=1)

m = len(sample)
d = len(words_dict.items())
X = np.zeros((m, d), np.float)
for i, s in enumerate(sample):
    for w in s:
       X[i][w] += 1 
lda.fit(X)
print np.sum(lda.components_,axis=1)
print lda.bound_
print lda.components_
