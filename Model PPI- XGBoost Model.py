#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc



pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 500)

# pip install xgboost


# ## Importing Data

# In[58]:


# import data
ppi_data = pd.read_excel(r'C://Users/Namrata/Desktop/Corporate Projects\Mortgage Insurance Cross Sell - Dataset & Case Description\Dataset.xls')


# ## Removing Redundant Variables

# In[59]:


# Remove redundant and forward looking variables having missing values
ppi_data.drop(columns = ['code','prdt_desc','category','Payment_Method','Insurance_Description','Ref','Telephone_Indicator','PPI_SINGLE','PPI_JOINT','PPI_LCI'], inplace = True)


# ## Outlier Detection & Treatment

# In[60]:


copy_ppi_data = ppi_data.copy()

col_list=['Credit_Score','Outstanding_Mortgage_Bal','APR','Value_of_Property','Total_value__CAIS_8_9s','Age','Mosaic','Term']

def outlier(col_list,df):
    for col in col_list:
        p99=np.percentile(df[col], 99)
        p1=np.percentile(df[col],1 )
        df.loc[ppi_data[col]>p99,col]=p99
        df.loc[ppi_data[col]<p1,col]=p1
    return df

outlier(col_list,ppi_data)


# ## Visualizations

# In[330]:


# sns.relplot(x='Age', y='Time_in_Employment', data=ppi_data, kind='scatter')
# plt.show()

# sns.set(rc={'figure.figsize':(5,5)})
# sns.kdeplot(ppi_data1['Credit_Score'],shade=True)

# sns.displot(ppi_data1['APR'])
# plt.title("APR")


# In[331]:


# fig, axes = plt.subplots(2,2, figsize=(10, 10)) 
# sns.countplot(ax=axes[0,0],x='PPI',hue='Final_Grade',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[0,1],x='PPI',hue='Gender',data=ppi_data,palette="mako")
# sns.countplot(ax=axes[1,0],x='PPI',hue='Employment_Status',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[1,1],x='PPI',hue='Mosaic_Class',data=ppi_data,palette="mako")

# fig, axes = plt.subplots(2,2, figsize=(10, 10)) 
# sns.countplot(ax=axes[0,0],x='PPI',hue='Loan_Type',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[0,1],x='PPI',hue='Residential_Status',data=ppi_data,palette="mako")
# sns.countplot(ax=axes[1,0],x='PPI',hue='Marital_Status',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[1,1],x='PPI',hue='Income_Range',data=ppi_data,palette="mako")

# fig, axes = plt.subplots(2,2, figsize=(10, 10)) 
# sns.countplot(ax=axes[0,0],x='PPI',hue='ACCESS_Card',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[0,1],x='PPI',hue='VISA_Card',data=ppi_data,palette="mako")
# sns.countplot(ax=axes[1,0],x='PPI',hue='American_Express',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[1,1],x='PPI',hue='Diners_Card',data=ppi_data,palette="mako") 
# # 
# fig, axes = plt.subplots(2,2, figsize=(10, 10)) 
# sns.countplot(ax=axes[0,0],x='PPI',hue='CIFAS_detected',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[1,1],x='PPI',hue='Bankruptcy_Detected__SP_',data=ppi_data,palette="mako") 
# sns.countplot(ax=axes[1,0],x='PPI',hue='Current_Account',data=ppi_data,palette="mako") 


# ## Correlation (Spearman)- With Target Variable

# In[ ]:


ppi_data[ppi_data.columns[1:]].corr(method="spearman")['PPI'][:]


# ## Coorelation (Spearman)- With Independent Variables

# In[61]:


my_r = ppi_data.corr(method="spearman")

# Generate a mask for the upper triangle
mask = np.zeros_like(my_r, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#cmap = sns.diverging_palette(20, 220, n=400)

# Draw the heatmap with the mask and correct aspect ratio
ax = sns.heatmap(my_r, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);




# ## Remove Highly Correlated Variables

# In[62]:


# Remove highly correlated variables
ppi_data.drop(columns = ['Mosaic','Time_since_most_recent_outstandi','Total___outstanding_CCJ_s','Time_since_most_recent_Public_In','Total_value__Public_Info___CCJ__','Total_Outstanding_Balances','Bureau_Data___Monthly_Other_Co_R','Total_outstanding_balance__mortg'], inplace = True)


# ## Create Bins

# In[63]:


ppi_data.loc[ppi_data['Income_Range'].isin([0,1]), 'Income_Bin'] = "Low" 
ppi_data.loc[ppi_data['Income_Range'].isin([2,3,4]), 'Income_Bin'] = "Med" 
ppi_data.loc[ppi_data['Income_Range'].isin([5,6]), 'Income_Bin'] = "High"


# In[64]:


ppi_data.loc[ppi_data['Total___of_accounts'].isin([0,1,2,3,4,5]), 'Accounts_Bin'] = 1 
ppi_data.loc[ppi_data['Total___of_accounts'].isin([6,7,8,9]), 'Accounts_Bin'] = 2 

ppi_data.drop(columns=['Income_Range','Total___of_accounts'],inplace=True)


# ## Create Dummy Variables

# In[65]:


# create dummy variables for grade,home_ownership,verification_status,pymnt_plan,purpose,application_type
# ppi_data1=pd.get_dummies(data=ppi_data,columns=['Residential_Status','Cheque_Guarantee','Final_Grade','Loan_Type','Marital_Status','Employment_Status'])
ppi_data1=pd.get_dummies(data=ppi_data,columns=['Worst_status_L6m','Worst_CUrrent_Status','__of_status_3_s_L6m','Searches___Total___L6m',
                                                'Income_Bin','Accounts_Bin','Mosaic_Class','Worst_History_CT','Total___Public_Info___CCJ____ban',
                                                'Final_Grade','Gender','Employment_Status','Bankruptcy_Detected__SP_',
                                                'Loan_Type','Residential_Status','Marital_Status','Full_Part_Time_Empl_Ind',
                                                'Perm_Temp_Empl_Ind','Current_Account','ACCESS_Card','VISA_Card','American_Express',
                                                'Diners_Card','Cheque_Guarantee','Other_Credit_Store_Card','CIFAS_detected'])

ppi_data1.head()


# ## Create Training and Testing Datasets

# In[66]:


inputs=ppi_data1.drop('PPI',axis='columns')
target=ppi_data1['PPI']

# divide dataset in train test
x_train,x_test,y_train,y_test= train_test_split(inputs, target, test_size=0.2 , random_state=1, stratify=target)


# ## Variable Selection
# 

# #### Feature Importance

# In[67]:


import xgboost as xgb

model_feature_imp = xgb.XGBClassifier()
model_feature_imp.fit(x_train,y_train)
model_feature_imp.get_booster().get_score(importance_type='weight')


# #### RFE
# 

# In[71]:


from sklearn.feature_selection import RFE
rfe = RFE(estimator=xgb.XGBClassifier(), n_features_to_select=30)
rfe.fit(x_train,y_train)

pd.DataFrame(rfe.support_,index=x_train.columns,columns=['selected'])


# In[3]:


python._versio_


# In[70]:


pd.DataFrame(rfe.ranking_,index=x_train.columns,columns=['Rank']).sort_values(by='Rank',ascending=True)


# In[92]:


feature_list=['Credit_Score',
'Term',
'Net_Advance',
'APR',
'Time_at_Address',
'Time_in_Employment',
'Time_with_Bank',
'Value_of_Property',
'Outstanding_Mortgage_Bal',
'Age',
'Total_value__CAIS_8_9s',
'Total_outstanding_balance___excl',
'Worst_status_L6m_0',
'Income_Bin_Low',
'Accounts_Bin_1.0',
'Mosaic_Class_1',
'Total___Public_Info___CCJ____ban_0',
'Marital_Status_M',]


# ### Grid Search cv

# In[74]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

xgb_model = xgb.XGBClassifier()

parameters = {'nthread':[4],
              'objective':['binary:logistic'],
              'learning_rate': [0.05,0.1], #so called `eta` value
              'max_depth': [4,6],
              'min_child_weight': [4,6],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'n_estimators': [500,1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [143]
             }

gsearch = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=5,
                   scoring='roc_auc',
                   verbose=2, refit=True)

gsearch.fit(x_train[feature_list], y_train)

# gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
# best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
# print('Raw AUC score:', score)
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))


# In[76]:


#best parameters from gridsearchcv
gsearch.best_params_ , gsearch.best_score_


# ## Model 

# In[93]:


from xgboost import XGBClassifier

# fit model no training data
xgb_model = XGBClassifier(colsample_bytree=0.8,
                             learning_rate=0.05,
                             max_depth=4,
                             min_child_weight=4,
                             n_estimators=500,
                             nthread=4,
                             objective='binary:logistic',
                             seed=143,
                             silent=1,
                             subsample=0.8)
xgb_model.fit(x_train[feature_list],y_train)


# ## Performance Indicators

# In[94]:


# Confusion matrix for test dataset
y_pred=xgb_model.predict(x_test[feature_list])


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
# print("Report : ",classification_report(y_test, y_pred))


# In[95]:


# Confusion matrix for training dataset
y_train_pred=xgb_model.predict(x_train[feature_list])


fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred)
roc_auc_train = auc(fpr_train, tpr_train)
print(roc_auc_train)
print("Confusion Matrix: ",confusion_matrix(y_train, y_train_pred))
print ("Accuracy : ",accuracy_score(y_train,y_train_pred)*100)
# print("Report : ",classification_report(y_test, y_pred))


# ## Feature Importance

# In[96]:


#feature importance
xgb_model.get_booster().get_score(importance_type='weight')


# ## ROC Curve for Training and Testing Datasets

# In[97]:


# ROC Curve for training dataset
# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')
# plot roc curves
plt.plot(fpr_train, tpr_train, linestyle='--',color='orange', label='XG Boost-Training Dataset')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

# Plot positive sloped 1:1 line for reference
plt.plot([0,1],[0,1])

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[98]:


# ROC Curve for test dataset
# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')
# plot roc curves
plt.plot(fpr, tpr, linestyle='--',color='orange', label='XG Boost- Test Dataset')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

# Plot positive sloped 1:1 line for reference
plt.plot([0,1],[0,1])

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# ## Algorithm Selection

# In[61]:


# from numpy import mean
# from numpy import std
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.pipeline import Pipeline
# from matplotlib import pyplot
 
# # get a list of models to evaluate
# def get_models():
# 	models = dict()
# 	# lr
# 	rfe = RFE(estimator=LogisticRegression(), n_features_to_select=15)
# 	model = DecisionTreeClassifier()
# 	models['lr'] = Pipeline(steps=[('s',rfe),('m',model)])
# 	# perceptron
# 	rfe = RFE(estimator=Perceptron(), n_features_to_select=15)
# 	model = DecisionTreeClassifier()
# 	models['per'] = Pipeline(steps=[('s',rfe),('m',model)])
# 	# cart
# 	rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=15)
# 	model = DecisionTreeClassifier()
# 	models['cart'] = Pipeline(steps=[('s',rfe),('m',model)])
# 	# rf
# 	rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=15)
# 	model = DecisionTreeClassifier()
# 	models['rf'] = Pipeline(steps=[('s',rfe),('m',model)])
# 	# gbm
# 	rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=15)
# 	model = DecisionTreeClassifier()
# 	models['gbm'] = Pipeline(steps=[('s',rfe),('m',model)])
# 	return models
 
# # evaluate a give model using cross-validation
# def evaluate_model(model, X, y):
# 	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# 	return scores
 
# # define dataset
# y = Y_train
# X = X_train

# # get the models to evaluate
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
# 	scores = evaluate_model(model, X, y)
# 	results.append(scores)
# 	names.append(name)
# 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()


# ### Insurance-wise Analysis
# 

# In[238]:


# import data
ppi_data_explore = pd.read_excel(r'C://Users/Namrata/Desktop/Tiger Analytics\Mortgage Insurance Cross Sell - Dataset & Case Description\Dataset.xls')

# Majority PPI holders have taken LASU or single.
insurance_wise=sns.countplot(x='Insurance_Description', data=ppi_data_explore)
insurance_wise.set_xticklabels(insurance_wise.get_xticklabels(), rotation=90, ha="right")
plt.tight_layout()
plt.show()


# In[257]:


sns.boxplot(x = ppi_data_explore["Age"], y = ppi_data_explore["Insurance_Description"]).set_title("Boxplot of Age by product type")

