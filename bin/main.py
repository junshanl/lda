import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from train import train_model, score

relevant_sample = []
irrelevant_sample = []

def to_word_list(context):   
   context = context.replace(',','')
   context = context.replace('.','')
   context = context.replace('\n','')
   return context.split(' ')

with open('./data/train/exam.txt', 'rb') as f:
   context = f.read()
   relevant_sample.extend(to_word_list(context))

with open('./data/train/relevant.txt') as f: 
   context = f.read()
   relevant_sample.extend(to_word_list(context))

with open('./data/train/irrelevant.txt') as f: 
   context = f.read()
   irrelevant_sample.extend(to_word_list(context))



def read_sample(path):
    with open(path) as f: 
       context = f.read()
       context = context.split('\n')
       lines = [to_word_list(line) for line in  context]
       X = [[words.index(w) for w in line if w in words] for line in lines]
    return X

model_xy, model_y, words = train_model(relevant_sample, irrelevant_sample)

def show_result(relevant_path, irrelevant_path):
    test_X=[]
    test_y=np.empty(0,)
    
    relevant_test = read_sample(relevant_path)
    irrelevant_test = read_sample(irrelevant_path)

    test_X.extend(relevant_test)
    test_X.extend(irrelevant_test)

    test_y = np.concatenate((test_y,np.ones(len(relevant_test))))
    test_y = np.concatenate((test_y,np.zeros(len(irrelevant_test))))

    print "probability:"
    scores = [score(model_xy,model_y, t) for t in test_X]
    print scores
    predict_y = np.round(scores) 
    print predict_y
    print classification_report(test_y, predict_y, target_names = ['relevant','irrelevant'])
    print "accuarcy:", accuracy_score(test_y, predict_y)



show_result('./data/train/relevant.txt','./data/train/irrelevant.txt')
show_result('./data/test2/relevant.txt','./data/test2/irrelevant.txt')
show_result('./data/test1/relevant.txt','./data/test1/irrelevant.txt')



