import keras
import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
from keras.models import load_model
model = load_model('Emotion_Det.h5')


data = pd.read_csv('../Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]
max_fatures = 2000
data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)

df_majority = data[data.sentiment=='Negative']
df_minority = data[data.sentiment=='Positive']
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=8000,    # to match majority class
                                 random_state=123) # reproducible results
data = pd.concat([df_majority, df_minority_upsampled])
data.sentiment.value_counts()
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')

twt = ['Meetings: Because we are lovely and are sick and sick of us.']
#vectorizing the tweet by the pre-fitted tokenizer instance
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")