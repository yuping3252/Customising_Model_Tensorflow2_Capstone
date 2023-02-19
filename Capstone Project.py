#!/usr/bin/env python
# coding: utf-8

# # Capstone Project
# ## Neural translation model
# ### Instructions
# 
# In this notebook, you will create a neural network that translates from English to German. You will use concepts from throughout this course, including building more flexible model architectures, freezing layers, data processing pipeline and sequence modelling.
# 
# This project is peer-assessed. Within this notebook you will find instructions in each section for how to complete the project. Pay close attention to the instructions as the peer review will be carried out according to a grading rubric that checks key parts of the project instructions. Feel free to add extra cells into the notebook as required.
# 
# ### How to submit
# 
# When you have completed the Capstone project notebook, you will submit a pdf of the notebook for peer review. First ensure that the notebook has been fully executed from beginning to end, and all of the cell outputs are visible. This is important, as the grading rubric depends on the reviewer being able to view the outputs of your notebook. Save the notebook as a pdf (File -> Download as -> PDF via LaTeX). You should then submit this pdf for review.
# 
# ### Let's get started!
# 
# We'll start by running some imports, and loading the dataset. For this project you are free to make further imports throughout the notebook as you wish. 

# In[2]:


import tensorflow as tf
import tensorflow_hub as hub
import unicodedata
import re
import pandas as pd
import numpy as np


# ![Flags overview image](data/germany_uk_flags.png)
# 
# For the capstone project, you will use a language dataset from http://www.manythings.org/anki/ to build a neural translation model. This dataset consists of over 200,000 pairs of sentences in English and German. In order to make the training quicker, we will restrict to our dataset to 20,000 pairs. Feel free to change this if you wish - the size of the dataset used is not part of the grading rubric.
# 
# Your goal is to develop a neural translation model from English to German, making use of a pre-trained English word embedding module.

# In[3]:


# Run this cell to load the dataset

NUM_EXAMPLES = 20000
data_examples = []
with open('data/deu.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        
        if len(data_examples) < NUM_EXAMPLES:
            data_examples.append(line)
        else:
            break


# In[4]:


# These functions preprocess English and German sentences

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"ü", 'ue', sentence)
    sentence = re.sub(r"ä", 'ae', sentence)
    sentence = re.sub(r"ö", 'oe', sentence)
    sentence = re.sub(r'ß', 'ss', sentence)
    
    sentence = unicode_to_ascii(sentence)
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-z?.!,']+", " ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    return sentence.strip()
    


# #### The custom translation model
# The following is a schematic of the custom translation model architecture you will develop in this project.
# 
# ![Model Schematic](data/neural_translation_model.png)
# 
# Key:
# ![Model key](data/neural_translation_model_key.png)
# 
# The custom model consists of an encoder RNN and a decoder RNN. The encoder takes words of an English sentence as input, and uses a pre-trained word embedding to embed the words into a 128-dimensional space. To indicate the end of the input sentence, a special end token (in the same 128-dimensional space) is passed in as an input. This token is a TensorFlow Variable that is learned in the training phase (unlike the pre-trained word embedding, which is frozen).
# 
# The decoder RNN takes the internal state of the encoder network as its initial state. A start token is passed in as the first input, which is embedded using a learned German word embedding. The decoder RNN then makes a prediction for the next German word, which during inference is then passed in as the following input, and this process is repeated until the special `<end>` token is emitted from the decoder.

# # 1. Text preprocessing
# * Create separate lists of English and German sentences, and preprocess them using the `preprocess_sentence` function provided for you above.
# * Add a special `"<start>"` and `"<end>"` token to the beginning and end of every German sentence.
# * Use the Tokenizer class from the `tf.keras.preprocessing.text` module to tokenize the German sentences, ensuring that no character filters are applied. _Hint: use the Tokenizer's "filter" keyword argument._
# * Print out at least 5 randomly chosen examples of (preprocessed) English and German sentence pairs. For the German sentence, print out the text (with start and end tokens) as well as the tokenized sequence.
# * Pad the end of the tokenized German sequences with zeros, and batch the complete set of sequences into one numpy array.

# In[5]:


from tensorflow.keras.preprocessing.text import Tokenizer

# crate separate lists of English and German setences
c_examples = [s.split('\tCC-BY 2.0 (France) Attribution')[0] for s in data_examples]
c_pairs = [re.split("\t", s) for s in c_examples]

c_eng = [x[0].strip() for x in c_pairs]
c_deu = [x[1].strip() for x in c_pairs]

english_sentences = [preprocess_sentence(s) for s in c_eng]    # list of english sentences (strings)

# add a special "<start>" and "<end>" token to each of the German sentences.
german_sentences  = ['<start> ' + preprocess_sentence(s) + ' <end>' for s in c_deu]


# In[6]:


# Use the Tokenizer class from the tf.keras.preprocessing.text module to tokenize the German sentences, 
# ensuring that no character filters are applied. Hint: use the Tokenizer's "filter" keyword argument.

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = None, filters = '', split = ' ', 
                      char_level = False, oov_token='<OOV>')
tokenizer.fit_on_texts(german_sentences)

import json
tconfig = tokenizer.get_config()
index_word = json.loads(tconfig['index_word'])
word_index = json.loads(tconfig['word_index'])
print(max(word_index.values()))


# In[9]:


gtokens    = []
gsequences = []
for i, s in enumerate(german_sentences):
    wordlst = s.split()
    gtokens.append(wordlst)                             # list of list(sentence) of german tokens (word)
    gsequences.append(np.squeeze(np.array(tokenizer.texts_to_sequences(wordlst))))  # list of ndarray of sequences (sentences)


# In[10]:


# print 5 random examples of English and German sentence pairs. 
# For German sentence, also print out the tokenized sequence

import numpy as np

print('English tokens:','-------------------','German tokens:', '------------------------------',  'german sequences')
idx = np.random.randint(0,20000, 5)
for i in idx:
    print(english_sentences[i],'----------', german_sentences[i], '----------', gsequences[i])


# In[11]:


# pad the end of each tokenized German sequences with zeros, and batch the whole set of sequences into a numpy array.

from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_german_seq = pad_sequences(gsequences, padding='post')   # ndarray (2000, 13)  int    <start> -> 2, <end> -> 3

pad_german_seq.shape


# ## 2. Prepare the data with tf.data.Dataset objects

# #### Load the embedding layer
# As part of the dataset preproceessing for this project, you will use a pre-trained English word embedding module from TensorFlow Hub. The URL for the module is https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1. This module has also been made available as a complete saved model in the folder `'./models/tf2-preview_nnlm-en-dim128_1'`. 
# 
# This embedding takes a batch of text tokens in a 1-D tensor of strings as input. It then embeds the separate tokens into a 128-dimensional space. 
# 
# The code to load and test the embedding layer is provided for you below.
# 
# **NB:** this model can also be used as a sentence embedding module. The module will process each token by removing punctuation and splitting on spaces. It then averages the word embeddings over a sentence to give a single embedding vector. However, we will use it only as a word embedding module, and will pass each word in the input sentence as a separate token.

# In[12]:


# Load embedding module from Tensorflow Hub
# embedding_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", 
embedding_layer  = hub.KerasLayer("./models/tf2-preview_nnlm-en-dim128_1", output_shape=[128], input_shape=[], dtype=tf.string)


# In[13]:


# Test the layer

embedding_layer(tf.constant(["these", "aren't", "the", "droids", "you're", "looking", "for"])).shape


# You should now prepare the training and validation Datasets.
# 
# * Create a random training and validation set split of the data, reserving e.g. 20% of the data for validation (NB: each English dataset example is a single sentence string, and each German dataset example is a sequence of padded integer tokens).
# * Load the training and validation sets into a tf.data.Dataset object, passing in a tuple of English and German data for both training and validation sets.
# * Create a function to map over the datasets that splits each English sentence at spaces. Apply this function to both Dataset objects using the map method. _Hint: look at the tf.strings.split function._
# * Create a function to map over the datasets that embeds each sequence of English words using the loaded embedding layer/model. Apply this function to both Dataset objects using the map method.
# * Create a function to filter out dataset examples where the English sentence is more than 13 (embedded) tokens in length. Apply this function to both Dataset objects using the filter method.
# * Create a function to map over the datasets that pads each English sequence of embeddings with some distinct padding value before the sequence, so that each sequence is length 13. Apply this function to both Dataset objects using the map method. _Hint: look at the tf.pad function. You can extract a Tensor shape using tf.shape; you might also find the tf.math.maximum function useful._
# * Batch both training and validation Datasets with a batch size of 16.
# * Print the `element_spec` property for the training and validation Datasets. 
# * Using the Dataset `.take(1)` method, print the shape of the English data example from the training Dataset.
# * Using the Dataset `.take(1)` method, print the German data example Tensor from the validation Dataset.

# In[14]:


# Create a random training and validation set split of the data, 
# reserving e.g. 20% of the data for validation 
# (NB: each English dataset example is a single sentence string, 
# and each German dataset example is a sequence of padded integer tokens).

import sklearn.model_selection as model_selection

english_train, english_test = model_selection.train_test_split(english_sentences,train_size=0.8) # list  len 16000   strings
german_train,  german_test  = model_selection.train_test_split(german_sentences, train_size=0.8) # list  len 16000   strings
g_token_train, g_token_test = model_selection.train_test_split(gtokens, train_size=0.8)          # list  len 16000   list strings
g_seq_train,   g_seq_test   = model_selection.train_test_split(pad_german_seq, train_size=0.8)   # ndarray (16000, 13)


# In[15]:


# Load the training and validation sets into a tf.data.Dataset object,
# passing in a tuple of English and German data for both training and validation sets.

dataset_train = tf.data.Dataset.from_tensor_slices((english_train, g_seq_train))  # TensorSliceDataset  (string, sequence)
dataset_test  = tf.data.Dataset.from_tensor_slices((english_test,  g_seq_test))


# In[16]:


# Create a function to map over the datasets that splits each 
# English sentence at spaces. 
# Apply this function to both Dataset objects using the map method.
# Hint: look at the tf.strings.split function.                    

def eng_split_(eng, deu):
    engsplitted = tf.strings.split(eng, " ")
    return engsplitted, deu

def split_func_(dataset):
    dataset_eng_splitted = dataset.map(eng_split_)
    return dataset_eng_splitted

dataset_eng_split_train = split_func_(dataset_train)   
dataset_eng_split_test  = split_func_(dataset_train)


# In[17]:


# Create a function to map over the datasets that embeds each sequence of English words
# using the loaded embedding layer/model. 
# Apply this function to both Dataset objects using the map method

ms1 = max(eng.shape[0] for eng, deu in dataset_eng_split_train)
ms2 = max(eng.shape[0] for eng, deu in dataset_eng_split_test)
ms  = max(ms1, ms2)
    
def embed_english_sentence(eng, deu):
    return embedding_layer(eng), deu

def embed_english_data(ms, dataset):
    return dataset.map(embed_english_sentence)

dataset_eng_embedded_train = embed_english_data(ms, dataset_eng_split_train)
dataset_eng_embedded_test  = embed_english_data(ms, dataset_eng_split_test)


# In[18]:


dataset_eng_embedded_train.element_spec


# In[19]:


# Create a function to filter out dataset examples where the English sentence
# is more than 13 (embedded) tokens in length. 
# Apply this function to both Dataset objects using the filter method.


def max13(eng, deu):
    m = tf.constant(14, dtype=tf.int32)    # filt out e with length more than 13, means keep e with length less than 14
    return tf.less(tf.shape(eng)[0],m)

def filt13(dataset):
    return dataset.filter(max13)

dataset_eng_filtered_train = filt13(dataset_eng_embedded_train)
dataset_eng_filtered_test  = filt13(dataset_eng_embedded_test)

print(dataset_eng_embedded_train.element_spec)


# In[20]:


# Create a function to map over the datasets that pads each English sequence of 
# embeddings with some distinct padding value before the sequence, so that each 
# sequence is length 13. 
# Apply this function to both Dataset objects using the map method. 
# Hint: look at the tf.pad function. 
# You can extract a Tensor shape using tf.shape; 
# you might also find the tf.math.maximum function useful.
#


def padeng(eng, deu):
    len1 = tf.shape(eng)[0]
    #eng = tf.reshape(tf.pad(eng, paddings=[[ms - len1, 0], [0,0]], mode='CONSTANT'), [13,128])
    eng = tf.pad(eng, paddings=[[13 - len1, 0], [0,0]], mode='CONSTANT')
    return eng, deu

dataset_eng_padded_train = dataset_eng_filtered_train.map(padeng)
dataset_eng_padded_test  = dataset_eng_filtered_test.map(padeng)

print(dataset_eng_padded_train.element_spec)


# In[21]:


# Batch both training and validation Datasets with a batch size of 16.

# english data: (16000, 13,128), for each on dim 0, a tensor, padding on the 1st dimension of it, padding value 0.0
# german data:each is a tokenized int sequences, one int per word, padded 0 to the end up to length 13

dataset_batch_train = dataset_eng_padded_train.batch(16)
dataset_batch_test  = dataset_eng_padded_test.batch(16)


# In[22]:


print(dataset_batch_train.element_spec)
print(dataset_batch_test.element_spec)


# In[23]:


# Using the Dataset .take(1) method, print the shape of the English data example 
# from the training Dataset and the validation dataset.

trainiter = iter(dataset_batch_train.take(1))
print("Shape of the English data example from the training dataset:")
print(trainiter.next()[0].shape)


# In[24]:


german_train = None
german_test  = None
g_token_train = None
g_token_test  = None 
dataset_eng_split_train    = None
dataset_eng_split_test     = None
dataset_eng_embedded_train = None
dataset_eng_embedded_test  = None
dataset_eng_filtered_train = None
dataset_eng_filtered_test  = None


# In[25]:


# Using the Dataset .take(1) method, print the shape of the German data example
# from the validation Dataset.

testiter = iter(dataset_batch_test.take(1))
print("Shape of the German data example from the validation dataset:")
print(testiter.next()[1])


# ## 3. Create the custom layer
# You will now create a custom layer to add the learned end token embedding to the encoder model:
# 
# ![Encoder schematic](data/neural_translation_model_encoder.png)

# You should now build the custom layer.
# * Using layer subclassing, create a custom layer that takes a batch of English data examples from one of the Datasets, and adds a learned embedded ‘end’ token to the end of each sequence. 
# * This layer should create a TensorFlow Variable (that will be learned during training) that is 128-dimensional (the size of the embedding space). _Hint: you may find it helpful in the call method to use the tf.tile function to replicate the end token embedding across every element in the batch._
# * Using the Dataset `.take(1)` method, extract a batch of English data examples from the training Dataset and print the shape. Test the custom layer by calling the layer on the English data batch Tensor and print the resulting Tensor shape (the layer should increase the sequence length by one).

# In[101]:


# Using layer subclassing, create a custom layer that takes a batch of English data examples
# from one of the Datasets, and adds a learned embedded ‘end’ token to the end of each sequence.
#
# This layer should create a TensorFlow Variable (that will be learned during training) that is
# 128-dimensional (the size of the embedding space). 
# Hint: you may find it helpful in the call method to use the tf.tile function to replicate 
# the end token embedding across every element in the batch.
#
# Understand the requirements:
# -- "takes a batch of"  batch_size=16, each sentence 13 tokens, each token embed to 128 dimension.
#    So, the tensor shape is (16, 13, 128)
# -- "adds a learned embedded 'end' token to the end of each sequence"
#    The "end of each sequence", sequence length is 13, added one more become a sequence of 14.
#    Pass ''end' to embedding_layer, that is straighforward understandable.
#    What about "add a learned embedded 'end' token" ?

from tensorflow.keras.layers import Layer, Input, LSTM

      
class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        embed_size = input_shape[-1]
        self.end   = self.add_weight(shape=(1, embed_size), initializer="random_normal")
        
        #self.embedend = tf.Variable(initial_value=tf.zeros([1,128]), trainable=True, dtype=tf.float32)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]        
        embed16 = tf.tile(tf.expand_dims(self.end, 0), [batch_size, 1, 1,])
        return tf.concat([inputs, embed16], 1)


# In[102]:


# Using the Dataset `.take(1)` method, extract a batch of English data examples from the training Dataset
# and print the shape. Test the custom layer by calling the layer on the English data batch Tensor and 
# print the resulting Tensor shape (the layer should increase the sequence length by one).

batchiter = iter(dataset_batch_train.take(1))
batch = batchiter.next()[0]

mylayer = MyLayer()
h = mylayer(batch)
print("MyLayer input  shape=", batch.shape)
print("MyLayer output shape=", h.shape)


# ## 4. Build the encoder network
# The encoder network follows the schematic diagram above. You should now build the RNN encoder model.
# * Using the functional API, build the encoder network according to the following spec:
#     * The model will take a batch of sequences of embedded English words as input, as given by the Dataset objects.
#     * The next layer in the encoder will be the custom layer you created previously, to add a learned end token embedding to the end of the English sequence.
#     * This is followed by a Masking layer, with the `mask_value` set to the distinct padding value you used when you padded the English sequences with the Dataset preprocessing above.
#     * The final layer is an LSTM layer with 512 units, which also returns the hidden and cell states.
#     * The encoder is a multi-output model. There should be two output Tensors of this model: the hidden state and cell states of the LSTM layer. The output of the LSTM layer is unused.
# * Using the Dataset `.take(1)` method, extract a batch of English data examples from the training Dataset and test the encoder model by calling it on the English data Tensor, and print the shape of the resulting Tensor outputs.
# * Print the model summary for the encoder network.

# In[103]:


# Using the functional API, build the encoder network according to the following spec:
#
# The model will take a batch of sequences of embedded English words as input, as given by the Dataset objects.
#
# The next layer in the encoder will be the custom layer you created previously, to add 
# a learned end token embedding to the end of the English sequence
#
# This is followed by a Masking layer, with the mask_value set to the distinct padding value
# you used when you padded the English sequences with the Dataset preprocessing above.
#
# The final layer is an LSTM layer with 512 units, which also returns the hidden and cell states.
#
# The encoder is a multi-output model. There should be two output Tensors of this model: 
# the hidden state and cell states of the LSTM layer. The output of the LSTM layer is unused.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, LSTM, Input

def encoderNet():   
    batch_size = 16
    num_tokens = 13
    dims       = 128 
    
    inputs = Input(batch_shape=(None, None, dims))
    h      = MyLayer()(inputs)
    h      = Masking(mask_value=0.0)(h)
    out_enc, h_enc, c_enc = LSTM(512, activation='tanh', return_state=True)(h)
    model = Model(inputs=inputs, outputs=[h_enc, c_enc], name="EncoderModel")
    return model


# In[104]:


encoder = encoderNet()
encoder.summary()

dataitr = iter(dataset_batch_train.take(1))
x = dataitr.next()[0]
h, c = encoder(x)
print('\n------------')
print("shape: from encoder, hidden state: ", h.shape)
print("shape: from encoder, cell state:   ", c.shape)


# You should now build the RNN decoder model.
# * Using Model subclassing, build the decoder network according to the following spec:
#     * The initializer should create the following layers:
#         * An Embedding layer with vocabulary size set to the number of unique German tokens, embedding dimension 128, and set to mask zero values in the input.
#         * An LSTM layer with 512 units, that returns its hidden and cell states, and also returns sequences.
#         * A Dense layer with number of units equal to the number of unique German tokens, and no activation function.
#     * The call method should include the usual `inputs` argument, as well as the additional keyword arguments `hidden_state` and `cell_state`. The default value for these keyword arguments should be `None`.
#     * The call method should pass the inputs through the Embedding layer, and then through the LSTM layer. If the `hidden_state` and `cell_state` arguments are provided, these should be used for the initial state of the LSTM layer. _Hint: use the_ `initial_state` _keyword argument when calling the LSTM layer on its input._
#     * The call method should pass the LSTM output sequence through the Dense layer, and return the resulting Tensor, along with the hidden and cell states of the LSTM layer.
# * Using the Dataset `.take(1)` method, extract a batch of English and German data examples from the training Dataset. Test the decoder model by first calling the encoder model on the English data Tensor to get the hidden and cell states, and then call the decoder model on the German data Tensor and hidden and cell states, and print the shape of the resulting decoder Tensor outputs.
# * Print the model summary for the decoder network.

# In[105]:


# Using Model subclassing, build the decoder network according to the following spec:
#  -- The initializer should create the following layers:
#         -- An Embedding layer with vocabulary size set to the number of unique German tokens, 
#            embedding dimension 128, and set to mask zero values in the input.
#         -- An LSTM layer with 512 units, that returns its hidden and cell states, and also returns sequences.
#         -- A Dense layer with number of units equal to the number of unique German tokens, and no activation function.


from tensorflow.keras.layers import Embedding, LSTM, Dense


class DecoderNet(Model):
    def __init__(self, word_index, **kwargs):
        super(DecoderNet, self).__init__(**kwargs)
        num_input_dim = max(word_index.values())+1
        
        self.embed = Embedding(input_dim = num_input_dim, output_dim = 128, mask_zero=True)        
        self.lstm  = LSTM(512, return_sequences=True, return_state=True)
        self.dense = Dense(num_input_dim)
        
    def call(self, inputs, h_state=None, c_state=None):
        h = self.embed(inputs)
        if h_state is not None and c_state is not None:
            outs, h2, c = self.lstm(h, initial_state=[h_state, c_state])
        else:
            outs, h2, c = self.lstm(h)
        outputs = self.dense(outs)
        return outputs, h2, c
  


# In[106]:


# Using the Dataset .take(1) method, extract a batch of English and German data examples 
# from the training Dataset. 
# Test the decoder model by first calling the encoder model on the English data Tensor 
# to get the hidden and cell states, and then call the decoder model on the German 
# data Tensor and hidden and cell states, and print the shape of the resulting decoder
# Tensor outputs.

dataiter = iter(dataset_batch_train.take(1))
eng, deu = dataiter.next()

encoder = encoderNet()
h, c = encoder(eng)

decoder = DecoderNet(word_index)
deco_out, deco_h, deco_c = decoder(deu, h, c)

print("input:    eng  shape           ", eng.shape)
print("input:    deu  shape           ", deu.shape)
print("output:   decoder output shape ", deco_out.shape)
print("output:   hidden state shape   ", deco_h.shape)
print("output:   cell state shape     ", deco_c.shape)


# In[107]:


decoder.summary()


# ## 6. Make a custom training loop
# You should now write a custom training loop to train your custom neural translation model.
# * Define a function that takes a Tensor batch of German data (as extracted from the training Dataset), and returns a tuple containing German inputs and outputs for the decoder model (refer to schematic diagram above).
# * Define a function that computes the forward and backward pass for your translation model. This function should take an English input, German input and German output as arguments, and should do the following:
#     * Pass the English input into the encoder, to get the hidden and cell states of the encoder LSTM.
#     * These hidden and cell states are then passed into the decoder, along with the German inputs, which returns a sequence of outputs (the hidden and cell state outputs of the decoder LSTM are unused in this function).
#     * The loss should then be computed between the decoder outputs and the German output function argument.
#     * The function returns the loss and gradients with respect to the encoder and decoder’s trainable variables.
#     * Decorate the function with @tf.function
# * Define and run a custom training loop for a number of epochs (for you to choose) that does the following:
#     * Iterates through the training dataset, and creates decoder inputs and outputs from the German sequences.
#     * Updates the parameters of the translation model using the gradients of the function above and an optimizer object.
#     * Every epoch, compute the validation loss on a number of batches from the validation and save the epoch training and validation losses.
# * Plot the learning curves for loss vs epoch for both training and validation sets.
# 
# _Hint: This model is computationally demanding to train. The quality of the model or length of training is not a factor in the grading rubric. However, to obtain a better model we recommend using the GPU accelerator hardware on Colab._

# In[108]:


# Define a function that takes a Tensor batch of German data (as extracted from the training Dataset), and 
# returns a tuple containing German inputs and outputs for the decoder model (refer to schematic diagram above).

def deu_input_output(deu):
    din  = deu[:, :-1]
    dout = deu[:,1:]
    return din, dout

deu_input, deu_output = deu_input_output(deu)
print("german sequence       =", deu)
print("deu_input             =", deu_input)
print("deu_output            =", deu_output)


# In[109]:


# Define a function that computes the forward and backward pass for your translation model. 
# This function should take an English input, German input and German output as arguments, 
# and should do the following:
#     -- Pass the English input into the encoder, to get the hidden and cell states of the encoder LSTM.
#     -- These hidden and cell states are then passed into the decoder, along with the German inputs, 
#        which returns a sequence of outputs (the hidden and cell state outputs of the decoder LSTM 
#        are unused in this function).
#     -- The loss should then be computed between the decoder outputs and the German output function argument.
#     -- The function returns the loss and gradients with respect to the encoder and decoder’s trainable variables.
#     -- Decorate the function with @tf.function

from tensorflow.keras.losses import SparseCategoricalCrossentropy

lossf = SparseCategoricalCrossentropy(from_logits=True)
optim = tf.keras.optimizers.Adam(learning_rate = 0.001)

from tensorflow.keras.losses import MSE

@tf.function
def fbpass(encoder, decoder, lossf, eng, deu):
    deu_in, deu_out = deu[:,:-1], deu[:,1:]
    with tf.GradientTape() as tape:
        h, c  = encoder(eng)        
        deco_outs, deco_h, deco_c = decoder(deu_in, h_state=h, c_state=c)      
        loss = tf.math.reduce_mean(lossf(deu_out, deco_outs))
        train_v = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(loss, train_v)
    return loss, grads


# In[110]:



def compute_loss(lossf, eng, deu):
    deu_in, deu_out = deu[:,:-1], deu[:,1:]
    enc_h, enc_c = encoder(eng)
    deco_outs, deco_h, deco_c = decoder(deu_in, h_state=enc_h, c_state=enc_c)
    return lossf(deu_out, deco_outs)
    


# In[111]:


epoch_h = {
    'epoch': [],
    'loss':  [],
    'val_loss': []
}
batch_h = {
    'id':   [],
    'loss': []
}


# In[1]:



# with 200 batches of data, each epoch runs about 50 minutes, and 5 epochs should 
# run around 5 hours, but the course colab environment is timed out in the middle 
# of 3rd epoch. Reduce the data to 100 batches, should be able to complete.
# with 100 batches for training, 30 batches for validation, still time out
# try 70 batches for training, 20 batches for validation

dataset_batch_short_train = dataset_batch_train.take(70)
dataset_batch_short_test  = dataset_batch_test.take(20)


# In[ ]:


# Define and run a custom training loop for a number of epochs (for you to choose) 
# that does the following:
#     -- Iterates through the training dataset, and creates decoder inputs and outputs
#        from the German sequences.
#     -- Updates the parameters of the translation model using the gradients of the function
#        above and an optimizer object.
#     -- Every epoch, compute the validation loss on a number of batches from the validation
#        and save the epoch training and validation losses.
#
# I am able to run this training in my home Ubuntu server with a fast 3.5GHz CPU with sufficient
# memory in less than 10 minutes.
#
# But, it is extremely slow in this Jupyter environment, too slow to complete. Also, the time
# window is limited, I am sure but I feel 2-3 hours maximum. After that, it automatically shut down
# and the result is lost.
#
# I tried multipel times. The program takes about 10-13 seconds to complete a batch of 16 data
# examples. Therefore, 16000 samples would need about 4 hours to run, exceeds the limit of the time
# window given in this jupyter environment.
#
# So, I implemented save_weights and load_weights, to continue to train the model to an acceptable
# loss value. But only the loss information of the last run will be displayed/printed/plotted



from timeit import default_timer as timer


def train_loop(encoder, decoder, lossf, optimizer, fbpass, 
               dataset_train, dataset_test, num_epochs=0):
    
    validation_steps = 30
    train_losses = []
    val_losses   = []

    for epoch in range(num_epochs):

        epochloss = tf.keras.metrics.Mean()
            
        tm_epoch_start   = timer()
        tm_batch10_start = timer()
            
        i = 0
        for eng, deu in dataset_train:
            loss, grads = fbpass(encoder, decoder, lossf, eng, deu)
            epochloss(loss)

            if i % 10 == 0 and i != 0:
                print(f"Epoch={epoch}, batch={i}, train, time={timer() - tm_batch10_start}, loss={loss}" )
                tm_batch10_start = timer()
                batch_h['id'].append(i)
                batch_h['loss'].append(loss)
                
            if i % 100 == 0 and i != 0:
                try:
                    encoder.save_weights("models/capstone_whts/encoder_weights.h5")
                    decoder.save_weights("models/capstone_whts/decoder_weights.h5")
                    print(f"Epoch={epoch}, batch={i}, weights saved")
                except:
                    print(f"Epoch={epoch}, batch={i}, cannot save weights *************")
            i += 1
            
            optim.apply_gradients(zip(grads, encoder.trainable_variables+decoder.trainable_variables))                      
        
        print(f"Epoch={epoch}, train,      loss={epochloss.result()},    epoch training   time ={timer() - tm_epoch_start}")  
        tm_epoch_val_start = timer()
        
        for eng, deu in dataset_test.take(validation_steps):
            val_loss = tf.keras.metrics.Mean()
            val_loss(compute_loss(lossf, eng, deu))  
        
        print(f"Epoch={epoch}, validation, val_loss={val_loss.result()}, epoch validation time ={timer() - tm_epoch_val_start}")
        
        epoch_h['epoch'].append(epoch)
        epoch_h['loss'].append(epochloss.result())
        epoch_h['val_loss'].append(val_loss.result())    
        print(f"Epoch={epoch}, train + validation, epoch time ={timer() - tm_epoch_start}")  

    return epoch_h

train_start_time = timer()
encoder = encoderNet()
decoder = DecoderNet(word_index)

try:
    print("start loading encoder weights from last run")
    encoder.load_weights("models/capstone_whts/encoder_weights.h5")
    print("done loading encoder weights from last run")
except:
    print("no previous encoder weights to load. OK")
try:
    print("start loading decoder weights from last run")
    decoder.load_weights("models/capstone_whts/decoder_weights.h5")
    print("done loading decoder weights from last run")
except:
    print("no previous decoder weights to load. OK")
   
print("training start ...")
h = train_loop(encoder, decoder, lossf, optim, fbpass,
               dataset_batch_short_train, dataset_batch_short_test, num_epochs=5)

print("running time=", timer() - train_start_time)


# In[83]:


# Plot the learning curves for loss vs epoch for validation sets.

import matplotlib.pyplot as plt

plt.plot(tloss)
plt.plot(vloss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss vs Epochs")
plt.xticks([0,5,10])
plt.legend(["Train", "Validation"])
plt.show()


# ## 7. Use the model to translate
# Now it's time to put your model into practice! You should run your translation for five randomly sampled English sentences from the dataset. For each sentence, the process is as follows:
# * Preprocess and embed the English sentence according to the model requirements.
# * Pass the embedded sentence through the encoder to get the encoder hidden and cell states.
# * Starting with the special  `"<start>"` token, use this token and the final encoder hidden and cell states to get the one-step prediction from the decoder, as well as the decoder’s updated hidden and cell states.
# * Create a loop to get the next step prediction and updated hidden and cell states from the decoder, using the most recent hidden and cell states. Terminate the loop when the `"<end>"` token is emitted, or when the sentence has reached a maximum length.
# * Decode the output token sequence into German text and print the English text and the model's German translation.

# In[ ]:





# In[ ]:





# In[ ]:




