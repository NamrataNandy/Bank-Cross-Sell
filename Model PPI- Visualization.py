#!/usr/bin/env python
# coding: utf-8

# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')

# import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 500)


# In[35]:


# import data
ppi_data = pd.read_excel(r'C://Users/Namrata/Desktop/Mortgage Insurance Cross Sell - Dataset & Case Description\Dataset.xls')


# In[36]:


#gives the count of records across variables
ppi_data.shape

# Describe function will only show non missing info.
ppi_data.describe()
ppi_data.isnull().sum()


# In[37]:


# Profiling of only PPI holders
ppi_holders=ppi_data[ppi_data['PPI']==1]

ppi_data["PPI"].value_counts()


# In[38]:


ppi_data.head()


# In[31]:


# Distribution of target variable- PPI
response = ppi_data.loc[:,"PPI"].value_counts().rename('Count')
plt.xlabel("PPI")
plt.ylabel('Count')
sns.barplot(response.index , Response.values,palette="mako")

fig, axes = plt.subplots(2, 2, figsize=(10, 10)) 
sns.countplot(ax=axes[0,0],x='PPI',hue='Gender',data=ppi_data,palette="mako") 
sns.countplot(ax=axes[0,1],x='PPI',hue='Final_Grade',data=ppi_data,palette="mako") 
sns.countplot(ax=axes[1,0],x='PPI',hue='Employment_Status',data=ppi_data,palette="mako") 
sns.countplot(ax=axes[1,1],x='PPI',hue='Bankruptcy_Detected__SP_',data=ppi_data,palette="mako")


# In[41]:


# ~75% of PPI holders have taken unsecured product
# sns.countplot(x='Loan_Type', data=ppi_data)
# plt.show()


# In[43]:


# using sns for plotting histogram of Credit_Score
# distribution is left skewed, centered at around 800-900, with most ppi holding customers with credit score 900, with apparent outliers. 
# Customers with good credit history are given PPI
sns.distplot(ppi_holders["Credit_Score"], kde = False).set_title("Histogram of Credit_Score")
plt.show()

# Profiling of non PPI holders
non_ppi_holders=ppi_data[ppi_data['PPI']==0]

# ppi_holders-Most of the customers are falling in 30-50 age group
sns.displot(ppi_holders['Age'])
# non-ppi holders- Most of the customers are falling in 30-50 age group
sns.displot(non_ppi_holders['Age'])


# In[6]:


# using sns for plotting histogram of Time_in_Employment
# distribution is right skewed, most ppi holding customers with time in employment 50 months, with apparent outliers. 
# Customers with initial employment duration have PPI. 
sns.distplot(ppi_holders["Time_in_Employment"], kde = False).set_title("Histogram of Time_in_Employment")
plt.show()


# In[7]:


# using sns for plotting histogram of Income_Range
# distribution is left skewed, majority ppi holding customers have income range >2. Customers with income range 6 have the purchase capability for PPL.
# Customers with initial employment duration have PPI. 
sns.distplot(ppi_holders["Income_Range"], kde = False).set_title("Histogram of Income_Range")
plt.show()


# In[8]:


# Age and time_in_employment have a relation
# using sns for creating scatter plot
sns.relplot(x='Age', y='Time_in_Employment', data=ppi_holders, kind='scatter')
plt.show()
# With age increasing, time in employment incraeses. Certain population fall in retirement bucket with age on higher side and no time in employment.


# In[9]:


# Outstanding_Mortgage_Bal and Total_Outstanding_Balances have a linear relation and have a positive correlation.
# using sns for creating scatter plot
sns.relplot(x='Outstanding_Mortgage_Bal', y='Total_Outstanding_Balances', data=ppi_holders, kind='scatter')
plt.show()


# In[25]:



sns.boxplot(x = ppi_holders["Age"], y = ppi_holders["Insurance_Description"]).set_title("Boxplot of Age by product type")

