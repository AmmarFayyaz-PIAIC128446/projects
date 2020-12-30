#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


x = np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


y = np.arange(10,50)


# 4. Find the shape of previous array in question 3

# In[4]:


np.shape(y)


# 5. Print the type of the previous array in question 3

# In[5]:


y.dtype


# 6. Print the numpy version and the configuration
# 

# In[6]:


np.__version__
np.show_config()


# 7. Print the dimension of the array in question 3
# 

# In[7]:


y.ndim


# 8. Create a boolean array with all the True values

# In[17]:


boolean_array = np.ones((2,2),dtype=np.bool) 
boolean_array


# 9. Create a two dimensional array
# 
# 
# 

# In[34]:


two = np.arange(1,11)
two.reshape(2,5)


# 10. Create a three dimensional array
# 
# 

# In[38]:


three = np.arange(16)
three.reshape(4,2,2)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[41]:


z = [1,1,1,0,2,2,2]
arr = np.array(z)
np.flip(arr)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[42]:


null = np.zeros(10)
null[5] = 1
null


# 13. Create a 3x3 identity matrix

# In[44]:


identity = np.identity(3)
identity


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[49]:


arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float32)
float_arr


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[11]:


import numpy as np
arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])

m = np.multiply(arr1,arr2)
m


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[12]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])

comparison = arr1 == arr2 
comparison


# 17. Extract all odd numbers from arr with values(0-9)

# In[71]:


arr = np.arange(10)
odd = np.where(arr%2 == 1)
odd


# 18. Replace all odd numbers to -1 from previous array

# In[75]:


arr = np.arange(10)
odd = np.where(arr%2 == 1,-1,arr)
odd


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[77]:


arr = np.arange(10)
arr[5] = 12
arr[6] = 12
arr[7] = 12
arr[8] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[89]:


arr = np.array((1,1,1,1,0,1,1,1,1))
arr.reshape(3,3)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[92]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1,1] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[108]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[137]:


two = np.array([[0,1,2,3,4],[5,6,7,8,9]])
two[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[139]:


two = np.array([[0,1,2,3,4],[5,6,7,8,9]])
two[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[156]:


two = np.array([[0,1,2,3,4],
                [5,6,7,8,9]])
two[0:2,2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[170]:


x = np.random.random((10,10))
min = x.min()
max = x.max()
print(min)
print(max)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[172]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[178]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
c = np.intersect1d(a, b)
c


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[50]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names != "Will"]


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[53]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names != 'Will,Joe']


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[83]:


import numpy as np
x = x = np.random.uniform(1,15,15)
x.reshape(5,3)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[87]:


x = x = np.random.uniform(1,16,16)
y = x.reshape(2,2,4)
y


# 33. Swap axes of the array you created in Question 32

# In[91]:


np.swapaxes(y,1,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[101]:


x = np.random.rand(10)
y = np.square(x)
z = np.where(y<0.5,0,y)
z


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[104]:


x = np.random.randint(12)
y = np.random.randint(12)
z = np.maximum(x,y)
z


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[115]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
x = np.unique(names)
x


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[118]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
a = np.delete(a,4)
a


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[173]:


import numpy as np
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
x = np.insert(sampleArray, 1, newColumn, axis=1)
x = np.delete(x,2)
x = np.delete(x,5)
x = np.delete(x,9)
x


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[120]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
z = np.dot(x,y) 
z


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[126]:


x = np.random.rand(20)
y = np.cumsum(x)
y


# In[ ]:




