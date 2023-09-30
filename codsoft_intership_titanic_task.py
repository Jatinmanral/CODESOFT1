# %% [markdown]
# Name:jatin manral
# 
# 
# 
# codsoft DS intership task 1
# 
# TITANIC SURVIVAL PREDICTION

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

# %%
df=pd.read_csv(r"C:\Users\jatin\OneDrive\Documents\python project\tested.csv")
df.head(10)

# %%
df.shape


# %%
df.describe()

# %%
df['Survived'].value_counts()

# %%
sns.countplot(x=df['Survived'],hue=df['Pclass'])


# %%
df['Sex']

# %%
sns.countplot(x=df["Sex"], hue=df["Survived"])

# %%
from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()
df["Sex"]=LabelEncoder.fit_transform(df['Sex'])
df.head

# %%
sns.countplot(x=df["Sex"], hue=df["Survived"])

# %%
df.isna().sum()

# %%
df=df.drop(["Age"],axis=1)

# %%


# %%
df_f=df
df_f.head()

# %%
x=df[['Pclass',"Sex"]]
y=df[["Survived"]]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %%
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(x_train,y_train)


# %%
p=print(log.predict(x_train))

# %%
print(y_test)

# %%
import warnings
warnings.filterwarnings('ignore')
res=log.predict([[1,0]])
if(res==0):
    print("dead")
else:
    print('survived')    

# %%



