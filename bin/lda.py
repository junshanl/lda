import numpy as np
import random

import os
from scipy.special import digamma, gammaln, gamma



class LDA():

    def __init__(self, n_components, method='em'):
        self.k = n_components
        self.method = method
        self.alpha = 1. / self.k
        self.eta = 1. / self.k
        pass
   
    def _random_topic_assign(self, X, d):
        z = []
        for x in X:
            z.append(np.array(np.round(np.random.rand(len(x),)*(d-1)), dtype=np.int))
        return z

    def _check_wordlist(self, X, dic):
        if dic is None:
            raise Exception('dic is not initialized')
        for x in X:
            for w in x:
                if not dic[w]:
                    raise Exception('{} not in dic'.format(x))


    def _init_mcmc_vars(self):
        self.Z = self.random_topic_assign(self.X, self.k)
        self.nmk = np.zeros((self.m, self.k), np.int)
        self.ndk = np.zeros((self.d, self.k), np.int)

        for i in range(self.m):
            for j in range(len(X)):
                t = self.Z[i][j]
                w = self.X[i][j]
                self.nmk[i][t]+=1
                self.ndk[w][t]+=1

        self.nmksum = np.sum(self.nmk, axis=1)
        self.ndksum = np.sum(self.ndk, axis=0)


    def fit(self, X, dic):
        self.m = len(X)
        self.d = len(dic)

        self.dic = dic
        self.X = X
        self._check_wordlist(self.X, self.dic)

        if self.method == 'em':

            self.lamb = self.em()
        elif self.method == 'mcmc':
            self._init_mcmc_vars(X)
        else:
            raise Exception('method should be em or mcmc')

    def transform(self, X):
        self._check_wordlist(X, self.dic)
        if self.lamb.shape == None:
            raise Exception('not trained lda ')

        doc_topic_distr, _  = self.e_step(X, self.lamb)
        return doc_topic_distr / np.sum(doc_topic_distr, axis=1)[:, np.newaxis]


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

    def e_step(self, X,  lamb):
        n_sample = len(X)
        gamma = np.random.gamma(100., 1./100., (n_sample, self.k))
        #gamma = np.ones((self.m, self.k))
        Elogbeta = self.dirichlet_expectation_2d(lamb)
        sstats = np.zeros((self.k, self.d))
       
        for i in range(100):
            last_gamma = gamma.copy()
            Elogtheta = self.dirichlet_expectation_2d(gamma)
           
            normphi = np.dot(np.exp(Elogtheta), np.exp(Elogbeta))

            for m, words in enumerate(X):
                phi = np.exp(Elogbeta[:, words] + Elogtheta[m,:, np.newaxis])
                phi = phi / normphi[m, words] 
                gamma[m] = self.alpha + np.sum(phi, axis=1)
            
            if np.sum(self.mean_change(last_gamma, gamma)) / self.k < 1e-3:
                break
        
        for m, words in enumerate(X):
            phi = np.exp(Elogbeta[:, words] + Elogtheta[m, :, np.newaxis])
            phi = phi / normphi[m, words]
            for i, w in enumerate(words):
                sstats[:, w] += phi[:, i]

        return gamma, sstats

    def m_step(self, sstats, lamb, n_iter):
        # rhot = np.power(self.learning_offset + n_iter, -self.learning_decay)
        # return lamb * (1-rhot) + rhot * (self.eta + self.m * sstats / self.d)
        return self.eta + sstats

    def approx_bound(self, X, gamma, lamb):
        score = 0
        Elogtheta = self.dirichlet_expectation_2d(gamma)
        Elogbeta = self.dirichlet_expectation_2d(lamb)

        # E[log p(docs | theta, beta)]
        '''
        for m, words in enumerate(X):
            phi = np.exp(Elogbeta[:, words] + Elogtheta[m, :, np.newaxis])
            score += np.sum(np.log(np.sum(phi, axis=0)))\
        '''
        normphi = np.dot(np.exp(Elogtheta), np.exp(Elogbeta))

        for m, words in enumerate(X):
            score += np.sum(np.log(normphi[m, words]))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self.alpha - gamma)*Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self.alpha))
        score += np.sum(gammaln(self.alpha*self.k) - gammaln(np.sum(gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += np.sum((self.eta - lamb)*Elogbeta)
        score += np.sum(gammaln(lamb) - gammaln(self.eta))
        score += np.sum(gammaln(self.eta*self.d) - gammaln(np.sum(lamb, 1)))
        return score


    def em(self):
        lamb = np.random.gamma(100., 1./100., (self.k, self.d)) 

        for n_iter in range(40):
            gamma, sstats = self.e_step(self.X, lamb)
            lamb = self.m_step(sstats, lamb, n_iter)
            bound = self.approx_bound(self.X, gamma, lamb)
            print "llh for ELBO: {}".format(bound)

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

    def score_em(self, X):
        pass

    def score_mcmc(self, X, Z):
        pass
'''
    def _save_model(self, path):
        with open(os.path.join(path,"Z.csv"), 'wb') as f:
            np.savetxt(f, self.Z, delimiter=",", fmt='%5s')
        with open(os.path.join(path,"W.csv"), 'wb') as f:
            np.savetxt(f, self.W, delimiter=",", fmt='%5s')

        with open(os.path.join(path,"nmk.csv"), 'wb') as f:
            np.savetxt(f, self.nmk, delimiter=",", fmt='%5s')
        with open(os.path.join(path,"ndk.csv"), 'wb') as f:
            np.savetxt(f, self.ndk, delimiter=",", fmt='%5s')

'''