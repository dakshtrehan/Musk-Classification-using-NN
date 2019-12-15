#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[13]:


x= pd.read_csv("musk_csv.csv", index_col=False)
x=x.drop(columns=["molecule_name", "conformation_name","ID"])


# In[14]:


print(x)


# In[15]:


x["class"].value_counts()


# In[16]:


x=np.array(x)


# In[17]:


x


# In[18]:


X = x[:,1:166]
Y = x[:,166]


# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[20]:


print(x_train.shape)


# In[119]:


model = Sequential()
model.add(Dense(25, input_dim=165, activation='relu')) 
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))  #Regularization
model.add(Dense(1, activation='sigmoid')) 


# In[120]:


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[121]:


b=model.fit(x_train, y_train, epochs = 10, batch_size=32, validation_data=(x_test, y_test))


# In[122]:


hist=b.history
print(hist)


# In[123]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(hist["val_accuracy"],label="Validation Accuracy")
plt.plot(hist["accuracy"],label="Training Accuracy")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.show()


# In[124]:


plt.plot(hist["val_loss"],label="Validation loss")
plt.plot(hist["loss"],label="Training loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# In[125]:


model.evaluate(x_test,y_test)


# In[126]:


model.evaluate(x_train,y_train)


# In[128]:


model.save("musk classification")


# In[ ]:




