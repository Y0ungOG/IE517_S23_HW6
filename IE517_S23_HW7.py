import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('//ad.uillinois.edu/engr-ews/chenyim2/Desktop/ccdefault.csv')
df = df.drop(df.columns[0], axis = 1)

df.head(-5)

#%%
#Use train_test split to see the performance
from sklearn.model_selection import train_test_split
df.columns
x = df.iloc[:,:-1].values
y = df['DEFAULT'].values
result_in = []
result_out = []
n_estimators_values = [5,50,100,200,500]
    #use pipeline to perform combined estimators
for n_estimator in n_estimators_values:
    x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)
    pipe_lr = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimator))
    pipe_lr.fit(x_train, y_train)
    y_pred = pipe_lr.predict(x_test)
    in_sample_score = pipe_lr.score(x_train, y_train)
    out_of_sample_score = pipe_lr.score(x_test, y_test)
    result_in.append((n_estimator, in_sample_score))
    result_out.append((n_estimator, out_of_sample_score))
    
for results in result_in :
    print(f"Random state: {results[0]}, Score: {results[1]}")
for results in result_out :
    print(f"Random state: {results[0]}, Score: {results[1]}")

#calculate the mean 
result_df_in = pd.DataFrame(result_in)
mean_in = np.mean(result_df_in[1])  
print(mean_in)  
result_df_out = pd.DataFrame(result_out)
mean_out = np.mean(result_df_out[1])  
print(mean_out)  

#table
table1 = pd.DataFrame({
    'Metric':[ 'Mean in-sample train-test score', 'Std in-sample train-test score', 'Mean out-sample train-test score', 'Std out-sample train-test score'],
    'value':[mean_in, result_df_in[1].std(), mean_out, result_df_out[1].std()]})

#%% 
#For individual fold accuracy score, mean CV score and std. use n_estimator = 5 to test processing time and accuracy
model = RandomForestClassifier(n_estimators=5)
score_ind = cross_val_score(model, x, y, cv = 10, scoring = 'accuracy')
print('Individual fold score', score_ind)
print('Individual CV accuracy: %.3f +/- %.3f'%(np.mean(score_ind),np.std( score_ind)))
  
#For individual fold accuracy score, mean CV score and std. use n_estimator = 200 to test processing time and accuracy
model = RandomForestClassifier(n_estimators=200)
score_ind = cross_val_score(model, x, y, cv = 10, scoring = 'accuracy')
print('Individual fold score', score_ind)
print('Individual CV accuracy: %.3f +/- %.3f'%(np.mean(score_ind),np.std( score_ind)))


# use k-fold cross-validation, calculate out of sample accuracy score
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True).split(x_train, y_train)

from sklearn.model_selection import cross_val_score
pipe_lr = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200))
scores = cross_val_score(estimator = pipe_lr, X = x_train, y = y_train, cv = 10, n_jobs =1)
print('Out of sample CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

#table
table2 = pd.DataFrame({
    'Metric':['Mean individual CV score', 'Std individual fold score','Mean out-sample CV score', 'Std out-sample fold score'],
    'value':[score_ind.mean(), score_ind.std(), scores.mean(), scores.std()]})



#%%  
#feature importance
feat_labels = df.columns[1:]

forest  = RandomForestClassifier(n_estimators = 500, random_state=42)
forest.fit(x_train,y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(x_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()


print("My name is {Chenyi Mao}")
print("My NetID is: chenyim2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
















