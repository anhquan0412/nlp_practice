#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self,e_word):
        super().__init__()
        self.e_word = e_word
        self.w_proj = nn.Linear(e_word,e_word)
        self.w_gate = nn.Linear(e_word,e_word)
        # init linear weight and bias?
    def forward(self,x_conv_out):
        """
         
        raw_input: (max_sentence_length,bs,max_word_length aka m)
        which should be output of to_input_tensor_char()
        --char_emb()-->
        x_emb: (max_sentence_length,bs,max_word_length,e_char)
        with e_char is size of character embedding. 
        
        --reshape()-->
        x_reshaped: (max_sentence_length,bs,e_char,max_word_length)
        
        --cnn()-->
        x_conv: (max_sentence_length,bs,e_word,max_word_length-k+1)
        with k is kernel size,e_word is the desired word embedding size
        TODO: do a loop for each sentence?
        
        --relu_and_globalmaxpool()-->
        x_conv_out: (max_sentence_length,bs,e_word)
        
        --high_way()-->
        x_highway: (max_sentence_length,bs,e_word)
        
        --dropout()-->
        x_word_emb: (max_sentence_length,bs,e_word)
        
        input: x_conv_out shape (bs,max_sentence_length,e_word)
        output: x_highway shape (bs,max_sentence_length,e_word) (no dropout applied)
        """
        
        x_proj = F.relu(self.w_proj(x_conv_out))
        x_gate = torch.sigmoid(self.w_gate(x_conv_out))
        x_highway = x_gate * x_proj + (1-x_gate) * x_conv_out
        return x_highway
