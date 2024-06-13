#!/usr/bin/env python
# coding: utf-8

# In[41]:


pip install ntscraper


# In[45]:


import pandas as pd
from ntscraper import Nitter
scraper = Nitter(log_level = 1,skip_instance_check = False)

def create_tweets_dataset(username,no_of_tweets):
  tweets = scraper.get_tweets(username,mode="user",number=no_of_tweets)
  data = {
    'link':[],
    'text':[],
    'user':[],
    'likes':[],
    'quotes':[],
    'retweets':[],
    'comments':[],
  }

  for tweet in tweets['tweets']:
    data['link'].append(tweet['link'])
    data['text'].append(tweet['text'])
    data['user'].append(tweet['user']['name'])
    data['likes'].append(tweet['stats']['likes'])
    data['quotes'].append(tweet['stats']['quotes'])
    data['retweets'].append(tweet['stats']['retweets'])
    data['comments'].append(tweet['stats']['comments'])

  df = pd.DataFrame(data)

  df.to_csv(username+"_tweets_data.csv")
  df.head()


# In[16]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# In[46]:


create_tweets_dataset("elonmusk",100)


# In[48]:


from pprint import pprint
elon_musk_info = scraper.get_profile_info(username="elonmusk")
pprint(elon_musk_info)


# In[1]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# In[2]:


import nltk
nltk.download('punkt')


# In[21]:


get_ipython().system('pip install clean-text')


# In[39]:


pip install seaborn


# In[4]:


get_ipython().system('pip install textblob')


# In[5]:


from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# In[ ]:





# In[6]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# In[ ]:





# In[9]:


import pandas as pd

df = pd.read_csv(r'C:\Users\chikw\elonmusk_tweets_data.csv')


# In[10]:


import pandas as pd

df = pd.read_csv('C:\\Users\\chikw\\elonmusk_tweets_data.csv')


# In[42]:


df.head()


# In[43]:


df.info()


# In[44]:


df.isnull().sum()


# In[45]:


df.columns


# In[ ]:





# In[15]:


text_df = df.drop(['link', 'user', 'likes', 'quotes', 'retweets','comments'], axis=1)
text_df.head()


# In[ ]:





# In[16]:


df_cleaned = text_df.dropna()


# In[17]:


df_cleaned.head(100)


# In[19]:


get_ipython().system('pip install clean-text')


# In[22]:


from cleantext import clean


# In[24]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[25]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

# Define a clean function to remove emojis (assuming it removes emojis)
def clean(text, no_emoji=True):
    if no_emoji:
        text = text.encode('ascii', 'ignore').decode('ascii')
    return text

# Define the data_processing function
def data_processing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"https\S+|www\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

# Load your DataFrame (replace with your actual file path)
text_df = pd.read_csv(r'C:\Users\chikw\elonmusk_tweets_data.csv')

# Apply the data_processing function to the 'text' column
text_df['text'] = text_df['text'].apply(data_processing)

# Display the first few rows to verify
print(text_df.head())


# In[26]:





# In[27]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

# Initialize the stemmer
stemmer = PorterStemmer()

# Define a clean function to remove emojis (assuming it removes emojis)
def clean(text, no_emoji=True):
    if no_emoji:
        text = text.encode('ascii', 'ignore').decode('ascii')
    return text

# Define the data_processing function
def data_processing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"https\S+|www\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    stemmed_text = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmed_text)

# Load your DataFrame (replace with your actual file path)
text_df = pd.read_csv(r'C:\Users\chikw\elonmusk_tweets_data.csv')

# Apply the data_processing function to the 'text' column
text_df['text'] = text_df['text'].apply(data_processing)

# Display the first few rows to verify
print(text_df.head())


# In[30]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


# In[46]:


text_df.head(10)


# In[ ]:





# In[41]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the DataFrame from the CSV file
text_df = pd.read_csv(r'C:\Users\chikw\elonmusk_tweets_data.csv')

# Create a figure with the specified size
plt.figure(figsize=(7, 7))

# Create the histogram using Seaborn
sns.countplot(x='sentiment', data=text_df, palette=["yellowgreen", "gold", "red"], edgecolor='black')

# Set the title and labels
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Counts')

# Display the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




