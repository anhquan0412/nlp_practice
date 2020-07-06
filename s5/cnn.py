#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,e_char,e_word,k=5,padding=1):
        super().__init__()
        self.conv1d = nn.Conv1d(e_char, e_word, kernel_size = k, padding = padding)
        self.mp1d = nn.AdaptiveMaxPool1d(1)
    def forward(self,x_reshaped):
        """
        input: x_reshaped: (max_sentence_length,bs,e_char,max_word_length)
        
        output:  x_conv_out: (max_sentence_length,bs,e_word)
            - e_word is the desired word embedding size
        """
        
        x_conv_out = []
        for each_sen in torch.split(x_reshaped,1,dim=0):
            each_sen = each_sen.squeeze(dim=0)
            
            x_conv = self.conv1d(each_sen) # (bs,e_word,max_word_length-k+1). 
            #relu
            result = F.relu(x_conv) # (bs,e_word,max_word_length-k+1)
            #maxpool
            result = self.mp1d(result).squeeze(2) # (bs,e_word,1) to (bs,e_word) after squeezing
            
            x_conv_out.append(result)
            
        x_conv_out = torch.stack(x_conv_out,dim=0)
        
        return x_conv_out