#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')


# In[2]:


default = pd.read_csv(r'D:\Simplilearn all projects\Data\default.csv')


# In[3]:


default.head()


# In[4]:


default.shape


# In[5]:


default = default.drop(['Index'],axis=1)


# In[6]:


np.round(default.describe(),2)


# In[7]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.boxplot(y=default['Bank Balance'])
plt.subplot(1,2,2)
sns.boxplot(y=default['Annual Salary'])
plt.show()


# In[8]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(default['Employed'])
plt.subplot(1,2,2)
sns.countplot(default['Defaulted?'])
plt.show()


# In[9]:


default['Employed'].value_counts()


# In[10]:


default['Defaulted?'].value_counts()


# In[11]:


default['Employed'].value_counts(normalize=True)


# In[12]:


default['Defaulted?'].value_counts(normalize=True)


# ### Bivariate Analyze

# In[13]:


plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
sns.boxplot(default['Defaulted?'],default['Bank Balance'])
plt.subplot(1,2,2)
sns.boxplot(default['Defaulted?'],default['Annual Salary'])


# In[14]:


pd.crosstab(default['Employed'], default['Defaulted?'], normalize='index').round(2)


# In[15]:


sns.heatmap(np.round(default[['Bank Balance','Annual Salary']].corr(),2), annot=True)


# In[16]:


default.isna().sum()


# ### Treating with the outliers

# In[17]:


Q1,Q3 = default['Bank Balance'].quantile([.25,.75])
IQR = Q3-Q1
LL = Q1-1.5*(IQR)
UL = Q3+1.5*(IQR)


# In[18]:


LL


# In[19]:


UL


# In[20]:


df = default[default['Bank Balance']>UL]
df


# In[21]:


df['Bank Balance'].count()


# In[22]:


df['Defaulted?'].value_counts(normalize=True)


# In[23]:


df['Defaulted?'].value_counts()


# In[24]:


default['Bank Balance'] = np.where(default['Bank Balance'] > UL, UL, default['Bank Balance'])


# In[25]:


sns.boxplot(default['Annual Salary'])


# In[26]:


default.head(2)


# In[27]:


x = default.drop(['Defaulted?'],axis=1)
y = default['Defaulted?']


# In[28]:


y


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.25, random_state=32)


# In[30]:


print(x_train.shape)
print(x_test.shape)


# In[31]:


print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


# In[32]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=33, sampling_strategy=0.75)
x_res,y_res = sm.fit_resample(x_train,y_train)


# In[33]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_res,y_res)


# In[34]:


from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
accuracy_score(y_test,lr.predict(x_test))


# In[35]:


confusion_matrix(y_test,lr.predict(x_test))

