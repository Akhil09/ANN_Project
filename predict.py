import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import sys
from keras.models import load_model
import pickle
np.random.seed(1337)
BASE_DIR = ''
TEXT_DATA_DIR = BASE_DIR + 'Test_data/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.2
print('Processing text data set ...')
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label news_group to numeric id
labels = []  # list of label ids
for news_group in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, news_group)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[news_group] = label_id
        for post in sorted(os.listdir(path)):
            if post.isdigit():
                post_path = os.path.join(path, post)
                if sys.version_info < (3,):
                    f = open(post_path)
                else:
                    f = open(post_path, encoding='latin-1')
                texts.append(f.read())
                f.close()
                labels.append(label_id)

target_start = 0
target_stop = 10
target_predictions = np.arange(target_start, target_stop)
print('Found %s texts.' % len(texts))
for target in target_predictions:
    print('Target blog: ', target, '\n', texts[target][0:1000], '\n __________ \n')
tokenizer = pickle.load(open('tokenizer.001.p', 'rb'))
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_test = to_categorical(np.asarray(labels))
# returns a compiled model identical to the previous one
model = load_model('model9.h5')
print('Generating Predictions ...\n')
print('Sample ', target_stop - target_start, 'Predictions: \n')
predictions = model.predict_on_batch(x_test)
class_predictions = predictions.argmax(axis=-1)
for target in target_predictions:
    print('Correct Category:   ', [label for (label, v) in labels_index.items() if v == labels[target]][0])
    print('Predicted Category is: ', [label for (label, v) in labels_index.items() if v == class_predictions[target]][0],'\n')
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy of Disk-Loaded Model: %.2f%%" % (scores[1]*100))
