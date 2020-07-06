#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        aka e_word
        
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.e_char = 50
        self.char_emb = nn.Embedding(len(vocab.char2id),self.e_char,padding_idx=vocab.char_pad)
        self.highway = Highway(self.word_embed_size)
        self.cnn = CNN(self.e_char,self.word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x_padded):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param x_padded: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary
        @param x_word_emb: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        
        
#         raw_input x_padded: (max_sentence_length,bs,max_word_length aka m)
#             - each integer is an index into the character vocabulary
#             - this should be output of to_input_tensor_char()
        
#         --char_emb()-->
#         x_emb: (max_sentence_length,bs,max_word_length,e_char)
#             - with e_char is size of character embedding.      
        x_emb = self.char_emb(x_padded)
        
#         --reshape()-->
#         x_reshaped: (max_sentence_length,bs,e_char,max_word_length)
        x_reshaped = x_emb.permute(0,1,3,2)
    
#         --cnn()-->
#         x_conv: (max_sentence_length,bs,e_word,max_word_length-k+1)
#             - with k is kernel size,e_word is the desired word embedding size
#             - do a loop for each sentence
#         --relu_and_globalmaxpool()-->
#         x_conv_out: (max_sentence_length,bs,e_word)
        x_conv_out = self.cnn(x_reshaped)

#         --high_way()-->
#         x_highway: (max_sentence_length,bs,e_word)
        x_highway = self.highway(x_conv_out)
#         --dropout()-->
#         x_word_emb: (max_sentence_length,bs,e_word)
        x_word_emb = self.dropout(x_highway)
        return x_word_emb

        
        