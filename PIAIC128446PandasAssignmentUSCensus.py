#!/usr/bin/env python
# coding: utf-8

# In[109]:


import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


# In[5]:


inventory = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\inventory.csv')
states0 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states0.csv')
states1 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states1.csv')
states2 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states2.csv')
states3 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states3.csv')
states4 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states4.csv')
states5 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states5.csv')
states6 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states6.csv')
states7 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states7.csv')
states8 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states8.csv')
states9 = pd.read_csv(r'C:\Users\AA\Desktop\pandas\Assignment\states9.csv')


# In[102]:


data_frame= [states0,states1,states2,states3,states4,states5,states6,states7,states8,states9]

us_census = pd.concat(data_frame)

us_census['Income'] = us_census['Income'].replace('\$', '', regex=True)

pop_split = pd.DataFrame(us_census.GenderPop.str.split('_',1).tolist(),
                                 columns = ['malepop','femalepop'])
pop_split.head()
us_census.head()


# In[108]:


pop_split['malepop'] = pop_split['malepop'].replace('M', '', regex=True)
pop_split['femalepop'] = pop_split['femalepop'].replace('F', '', regex=True)
pop_split['malepop'] = pop_split['malepop'].astype(int)
pop_split.fillna(0)
pop_split.head()


# In[111]:


plt.scatter(pop_split.malepop,pop_split.femalepop)


# In[115]:


pop_split.duplicated()


# In[116]:


pop_split.drop_duplicates() 


# In[117]:


plt.scatter(pop_split.malepop,pop_split.femalepop)


# In[ ]:


-

