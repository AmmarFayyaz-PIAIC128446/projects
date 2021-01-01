#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[9]:


import numpy as np
a = np.arange(10)
a.reshape(2,5)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[6]:


import numpy as np
a = np.arange(10)
a = a.reshape(2,5)
b = np.ones(10)
b = b.reshape(2,5)
c = np.vstack((a,b))
print(c)


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[13]:


import numpy as np
a = np.array([0, 1, 2, 3, 4, 1, 1, 1, 1, 1])
b = np.array([5, 6, 7, 8, 9, 1, 1, 1, 1, 1])
np.hstack((a,b))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[21]:


import numpy as np
a =  np.random.rand(5,5)
a.flatten()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[ ]:





# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[31]:


import numpy as ny
a = np.random.rand(27)
a = a.reshape(3,3,3)
print(a)
b = a.ndim
print('dimension of array a are =',b)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[33]:


a = np.random.rand(5,5)
b = np.square(a)
print('random array a=', a)
print('square of array a=',b)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[36]:


import numpy as np
a = np.random.rand(5,6)
b = np.mean(a)
print('the mean of random 5*6 array a is=',b)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[37]:


np.std(a)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[38]:


np.median(a)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[40]:


a.transpose()


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[45]:


a = np.random.rand(4,4)
b = np.diagonal(a)
np.sum(b)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[46]:


np.linalg.det(a)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[49]:


a = np.random.rand(4,4)
fifth = np.percentile(a,5)
ninety = np.percentile(a,90)
print('The 5th percentile is=',fifth)
print('the 90th percentile is=',ninety)


# ## Question:15

# ### How to find if a given array has any null values?

# In[51]:


b = np.arange(10)
np.isnan(b)

