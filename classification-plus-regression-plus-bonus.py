

import pandas as pd
import numpy as np
from scipy.sparse import hstack,csr_matrix
from sklearn.externals import joblib
import pickle
df = pd.read_csv('bing_hack_training.csv')


# In[3]:

from sklearn.feature_extraction.text import TfidfVectorizer


# In[109]:

tdidf_summ = TfidfVectorizer(ngram_range=(1,2))
tdidf_sub = TfidfVectorizer(ngram_range=(1,2))


# In[110]:

summary_matrix = tdidf_summ.fit_transform(df.summary)
sub_matrix = tdidf_sub.fit_transform(df.title)


# In[73]:


pickle.dump(tdidf_summ,open("models/summary_tdidf_filtered_new.pkl",'wb'))
pickle.dump(tdidf_sub,open("models/sub_tdidf_filtered_new.pkl",'wb'))


# In[215]:

tdidf_sub = pickle.load(open("models/sub_tdidf_filtered.pkl",'rb'))
tdidf_summ = pickle.load(open("models/summary_tdidf_filtered.pkl",'rb'))


# In[59]:

summary_matrix = tdidf_summ.transform(df.summary)
sub_matrix = tdidf_sub.transform(df.title)


# In[111]:

summary_matrix


# In[112]:

sub_matrix


# In[113]:

major_sparse_matrix = hstack([sub_matrix,summary_matrix])


# In[114]:

major_sparse_matrix


# In[10]:

Y = df.topic_id


# In[11]:

complete_list = []
for each_row in df.authors.str.split(';'):
    complete_list = complete_list + [k.strip() for k in each_row]


# In[12]:

complete_list = np.unique(complete_list)


# In[162]:

pickle.dump(complete_list,open("objects/complete_author_list.pkl",'wb'))


# In[93]:

unigrams = TfidfVectorizer(stop_words=['.']).fit(df.summary).get_feature_names()


# In[100]:

import math
uni_post_list=np.zeros(shape=(df.shape[0],len(unigrams)))
for idx_out,each_row in enumerate(df.summary.str.replace('.',' ').str.split()):
    for idx,each_uni in enumerate(unigrams):
        if each_uni in each_row[:int(math.ceil(0.2*len(each_row)))]:
            uni_post_list[idx_out,idx] =1


# In[106]:

uni_post_list = csr_matrix(uni_post_list)


# In[135]:

authors_list = np.zeros(shape=(df.shape[0],complete_list.shape[0]))
import pdb
for idx_out,each_author_list in enumerate(df.authors.str.split(';')):
    for idx,each_author in enumerate(complete_list):
        if each_author in each_author_list:
            #pdb.set_trace()
            authors_list[idx_out,idx] = 1


# In[108]:

major_sparse_matrix


# In[139]:

summary_matrix


# In[138]:

uni_post_list


# In[137]:

major_sparse_matrix.multiply(uni_post_list)


# In[115]:

authors_list = hstack([csr_matrix(authors_list),major_sparse_matrix,uni_post_list])


# In[116]:

authors_list


# In[342]:

opt_model =joblib.load('models/logreg_f1.pkl')


# In[67]:

Y_pred_classification = vc.predict(authors_list)


# In[68]:

Y_pred_classification


# In[69]:

pd.DataFrame(Y_pred_classification,columns=['topic_id']).to_csv('classification_problem_vc.csv')


# In[135]:

df['topic_id'] = Y_pred_classification


# In[58]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


# In[115]:

clfm = LogisticRegression()


# In[152]:

f1_score(clfm.predict(authors_list),Y)


# In[157]:

f1_score(clfsvm.predict(authors_list),Y)


# In[24]:

from sklearn import cross_validation


# In[60]:

from sklearn import grid_search
from sklearn.metrics import confusion_matrix,accuracy_score


# In[116]:

param_logreg = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'class_weight':['auto',None] }


# In[117]:

opt_model = grid_search.GridSearchCV(clfm, param_logreg,n_jobs=-1,scoring='f1',cv=10)


# In[118]:

opt_model.fit(authors_list,Y)


# In[207]:

opt_model.best_score_


# In[64]:

opt_model.best_score_


# In[119]:

opt_model.best_score_


# In[200]:

opt_model.best_score_


# In[139]:

joblib.dump(opt_model,'models/logreg_f1_new.pkl')


# In[84]:

from sklearn.linear_model import LinearRegression
clf_reg = LinearRegression()


# In[120]:

Y_year = df.publication_year


# In[121]:

clf_reg.fit(authors_list,Y_year)


# In[101]:

from sklearn import  cross_validation
cross_validation.cross_val_score(clf_reg,authors_list,Y_year,cv=10)


# In[122]:

from sklearn import  cross_validation
cross_validation.cross_val_score(clf_reg,authors_list,Y_year,cv=10)


# In[124]:

np.round(clf_reg.predict(authors_list))[:50]


# In[103]:

accuracy_score(np.round(clf_reg.predict(authors_list)),Y_year)


# In[136]:

Y_regression = np.round(clf_reg.predict(authors_list))


# In[137]:

df['publication_year'] = Y_regression


# In[138]:

joblib.dump(clf_reg,'models/linreg_f1_new.pkl')


# In[140]:

df.to_csv('data_test_new.csv')


# In[ ]:

df.get(['record_id'])


# In[239]:

df = pd.read_csv('bing_hack_training.csv')


# In[ ]:

tdidf_summ = TfidfVectorizer(ngram_range=(1,2),min_df=0.02)
tdidf_sub = TfidfVectorizer()


# In[248]:

test_data= ['this is sentence1. this is sentence 2','this is also sentence one, this is sentence 2']


# In[243]:

def sentence_splitter(string):
    return string.split('.')


# In[253]:

tdidf_summ_sentence=TfidfVectorizer(ngram_range=(1,2),tokenizer=sentence_splitter)


# In[254]:

summary_matrix = tdidf_summ_sentence.fit_transform(df.summary)
sub_matrix = tdidf_sub.fit_transform(df.title)


# In[255]:

summary_matrix


# In[256]:

sub_matrix


# In[257]:

major_sparse_matrix = hstack([sub_matrix,summary_matrix])


# In[258]:

major_sparse_matrix


# In[259]:

Y = df.topic_id


# In[263]:

complete_list = []
for each_row in df.authors.str.split(';'):
    complete_list = complete_list + [k.strip() for k in each_row]
complete_list = np.unique(complete_list)


# In[264]:

authors_list = np.zeros(shape=(df.shape[0],complete_list.shape[0]))
import pdb
for idx_out,each_author_list in enumerate(df.authors.str.split(';')):
    for idx,each_author in enumerate(complete_list):
        if each_author in each_author_list:
            #pdb.set_trace()
            authors_list[idx_out,idx] = 1


# In[265]:

authors_list = hstack([csr_matrix(authors_list),major_sparse_matrix])


# In[266]:

authors_list


# In[267]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


# In[268]:

clfm = LogisticRegression()


# In[269]:

param_logreg = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'class_weight':['auto',None] }


# In[270]:

opt_model_bonus = grid_search.GridSearchCV(clfm, param_logreg,n_jobs=-1,scoring='f1',cv=10)


# In[272]:

opt_model_bonus.fit(authors_list,Y)


# In[301]:

est = opt_model_bonus.best_estimator_


# In[314]:

total_column_names = complete_list.tolist() + tdidf_sub.get_feature_names() + tdidf_summ_sentence.get_feature_names() 


# In[315]:

complete_list.shape[0]


# In[367]:

topic_document = {}
for idx,class_coef in enumerate(est.coef_):
    
    author_list = class_coef[:complete_list.shape[0]]
    title_list = class_coef[complete_list.shape[0]:len(tdidf_sub.get_feature_names())+complete_list.shape[0]]
    summary_list = class_coef[len(tdidf_sub.get_feature_names())+complete_list.shape[0]:len(tdidf_sub.get_feature_names())+complete_list.shape[0]+len(tdidf_summ_sentence.get_feature_names())]
    topic_document[idx] = []
    au_features = np.argsort(author_list)[-9:].reshape(3,3)
    title_features = np.argsort(title_list)[-30:].reshape(3,10)
    summ_features = np.argsort(title_list)[-24:].reshape(3,8)
    for each in [0,1,2]:
        each_list = [';'.join([complete_list[k] for k in au_features[each]])]
        each_list = each_list + [' '.join([tdidf_sub.get_feature_names()[k] for k in title_features[each]])]
        each_list = each_list + ['.'.join([tdidf_summ_sentence.get_feature_names()[k] for k in summ_features[each]])]
        topic_document[idx].append(each_list)


# In[374]:

for index in est.classes_:
    pd.DataFrame(topic_document[j],columns=['authors','title','summary']).to_csv('topic_'+str(index)+'.csv')


# In[382]:

authors_list.shape


# In[326]:

k = [3, 4, 5, 6, 7, 8, 9]


# In[328]:

np.arange(10)[np.arange(3).shape[0]:len(k)+np.arange(3).shape[0]+!]


# In[324]:

np.arange(3)


# In[359]:

confusion_matrix(Y_pred_classification,Y)


# In[24]:

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


# In[25]:

clf_logreg = LogisticRegression()


# In[26]:

clf =AdaBoostClassifier(clf_logreg,n_estimators=100)


# In[45]:

cross_validation.cross_val_score(clf,authors_list.toarray(),Y,cv=10)


# In[16]:

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[164]:

opt_model.best_params_


# In[117]:

clf_logreg = LogisticRegression(C=100)
clf_nb = MultinomialNB(alpha=0.1)
clf_knn = KNeighborsClassifier(n_neighbors=5,weights='distance')


# In[126]:

vc = VotingClassifier(estimators=[('lg', clf_logreg), ('nb', clf_nb),('knn', clf_knn)],voting='soft')


# In[131]:

parame = {'nb__alpha' : [0.1,0.2,0.5,0.7,0.9,1.0],'knn__weights': ['uniform', 'distance'], 'knn__n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],'lg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'lg__class_weight':['balanced',None]}


# In[132]:

grid = grid_search.GridSearchCV(estimator=vc, param_grid=parame, cv=10)


# In[133]:

grid.fit(authors_list,Y)


# In[119]:

clf_logreg =clf_logreg.fit(authors_list,Y)


# In[120]:

clf_nb = clf_nb.fit(authors_list,Y)


# In[121]:

clf_knn = clf_knn.fit(authors_list,Y)


# In[122]:

vc = vc.fit(authors_list,Y)


# In[123]:

cv = cross_validation.cross_val_score(vc,authors_list,Y,cv=10)


# In[124]:

cv.mean()


# In[29]:

cv.mean()


# In[36]:

cv.mean()


# In[53]:

cv.mean()


# In[30]:

cv_old = cross_validation.cross_val_score(clf_logreg,authors_list,Y,cv=10)


# In[31]:

cv_old.mean()


# In[39]:

from sklearn import grid_search
opt_mod_knn = grid_search.GridSearchCV(clf_knn, parameters_knn,n_jobs=-1,scoring='f1',cv=10)


# In[40]:

opt_mod_knn.fit(authors_list,Y)


# In[42]:

opt_mod_knn.best_params_


# In[37]:

parameters_knn = {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
parameters_nb = {'alpha' : [0.1,0.2,0.5,0.7,0.9,1.0]}


# In[43]:

opt_mod_nb = grid_search.GridSearchCV(clf_nb, parameters_nb,n_jobs=-1,scoring='f1',cv=10)
opt_mod_nb.fit(authors_list,Y)


# In[45]:

opt_mod_nb.best_params_


# In[54]:

joblib.dump(clf_logreg,'models/logreg_vc.pkl')


# In[55]:

joblib.dump(clf_nb,'models/nb_vc.pkl')


# In[56]:

joblib.dump(clf_knn,'models/knn_vc.pkl')


# In[57]:

joblib.dump(vc,'models/vc.pkl')


# In[73]:




# In[ ]:



