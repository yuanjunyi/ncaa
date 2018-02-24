
# coding: utf-8

# In[1]:


# what is this line all about?!? Answer in lecture 4
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


from numpy import *


# In[3]:


v=array([1,2,3,4])
v


# In[4]:


# a matrix: the argument to the array function is a nested Python list
M = array([[1, 2], [3, 4]])

M


# In[5]:


type(v), type(M)


# In[7]:


v.shape, M.shape


# In[8]:


M.dtype


# In[9]:


x=arange(100,110,2)

x


# In[11]:


a = array([[1, 2, 3], [4, 5, 6]])


# In[12]:


a.shape


# In[13]:


# using linspace, both end points ARE included
linspace(0, 10, 25)


# In[14]:


help(linspace)


# In[15]:


linspace(1,5,5)


# In[16]:


help(logspace)


# In[17]:


logspace(1,5,5,base=10)


# In[18]:


help(mgrid)


# In[20]:


help(genfromtxt)


# In[22]:


data_dir='../DataFiles/'
cities = genfromtxt(data_dir+'Cities.csv', delimiter=',')
cities


# In[24]:


cities.shape


# In[26]:


M=random.rand(3,3)


# In[27]:


M


# In[28]:


savetxt('random_mat.txt', M)


# In[31]:


get_ipython().system('cat random_mat.txt')


# In[30]:


get_ipython().system('cat random_mat.txt')


# In[32]:


M[0]


# In[34]:


M[:,0]


# In[35]:


M[1,:]=1


# In[36]:


M


# In[37]:


A=array([1,2,3,4,5])


# In[38]:


A


# In[40]:


A[::2] *= -1


# In[41]:


A


# In[42]:


A[3:]


# In[43]:


A


# In[44]:


M


# In[45]:


M[:1,:1]


# In[46]:


M[:2,:1]


# In[47]:


B=array([n for n in range(5)])


# In[49]:


mask=array([True,False,True,False,False])


# In[50]:


B[mask]


# In[51]:


x=arange(0,10,0.5)


# In[52]:


mask=(5<x)*(x<7.5)


# In[53]:


x[mask]


# In[57]:


A=array([[n + m*10 for n in arange(5)] for m in arange(5)])
A


# In[58]:


A*A


# In[59]:


v1 = arange(0, 5)


# In[60]:


v1


# In[61]:


v1.shape


# In[62]:


v1[:0]


# In[63]:


v2=array([[n] for n in range(5)])
v2.shape


# In[64]:


A*v1


# In[65]:


A*v2


# In[69]:


M=matrix(A)
V=matrix(v1).T


# In[70]:


V


# In[71]:


M


# In[72]:


a=array([[1,2],[3,4]])
b=array([[5,6]])
a.shape,b.shape


# In[73]:


concatenate((a,b),axis=0)


# In[75]:


concatenate((a,b.T),axis=1)

