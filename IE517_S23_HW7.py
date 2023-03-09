import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import cross_val_score

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
n_estimators_values = [5]

for random_state in range(1,11):
    x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state = random_state)
    #use pipeline to perform combined estimators
    for n_estimator in n_estimators_values:
        pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), RandomForestClassifier(n_estimator))
        pipe_lr.fit(x_train, y_train)
        y_pred = pipe_lr.predict(x_test)
        in_sample_score = pipe_lr.score(x_train, y_train)
        out_of_sample_score = pipe_lr.score(x_test, y_test)
        result_in.append((random_state, in_sample_score))
        result_out.append((random_state, out_of_sample_score))
    
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
#For individual fold accuracy score, mean CV score and std 
model = DecisionTreeClassifier(random_state=42)
score_ind = cross_val_score(model, x, y, cv = 10, scoring = 'accuracy')
print('Individual fold score', score_ind)
print('Individual CV accuracy: %.3f +/- %.3f'%(np.mean(score_ind),np.std( score_ind)))


# use k-fold cross-validation, calculate out of sample accuracy score
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True).split(x_train, y_train)

from sklearn.model_selection import cross_val_score
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), DecisionTreeClassifier(random_state=42))
scores = cross_val_score(estimator = pipe_lr, X = x_train, y = y_train, cv = 10, n_jobs =1)
print('Out of sample CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

#table
table2 = pd.DataFrame({
    'Metric':['Mean individual CV score', 'Std individual fold score','Mean out-sample CV score', 'Std out-sample fold score'],
    'value':[score_ind.mean(), score_ind.std(), scores.mean(), scores.std()]})

print("My name is {Chenyi Mao}")
print("My NetID is: chenyim2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")























