
# coding: utf-8

# ## Play pandas and numpy libraries

# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


from IPython.display import display, Math, Latex


# In[5]:


display(Math(r'\sqrt{a^2 + b^2}')) 


# In[6]:


get_ipython().run_line_magic('lsmagic', '')


# In[11]:


get_ipython().run_line_magic('time', 'x = range(100)')


# In[8]:


echo


# In[14]:


get_ipython().system('pip list')


# In[17]:


from ipywidgets import widgets
from IPython.display import display

text=widgets.Text()
display(text)

def handle_submit(sender):
    print(text.value)
    
text.on_submit(handle_submit)

