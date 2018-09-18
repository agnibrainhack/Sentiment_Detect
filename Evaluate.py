import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model
model = load_model('Emotion_Det.h5')

twt = ['Meetings: Because we are lovely and are sick and sick of us.']
#vectorizing the tweet by the pre-fitted tokenizer instance
tokenizer.fit_on_texts(twt)
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")