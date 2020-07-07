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
        self.e_word = e_word
    def forward(self,x_reshaped):
        """
        input: x_reshaped: (max_sentence_length,bs,e_char,max_word_length)
        
        output:  x_conv_out: (max_sentence_length,bs,e_word)
            - e_word is the desired word embedding size
        """
#         x_conv_out2 = []
#         for each_sen in torch.split(x_reshaped,1,dim=0):
#             each_sen = each_sen.squeeze(dim=0) # bs,e_char,max_word_length
            
#             x_conv = self.conv1d(each_sen) # (bs,e_word,max_word_length-k+1). 
#             #relu
#             result = F.relu(x_conv) # (bs,e_word,max_word_length-k+1)
#             #maxpool
#             result = self.mp1d(result).squeeze(2) # (bs,e_word,1) to (bs,e_word) after squeezing
            
#             x_conv_out2.append(result)
            
#         x_conv_out2 = torch.stack(x_conv_out2,dim=0)
#         return x_conv_out2


        # you can combine first and second dimension to avoid loop while conv1d
        sent_length,bs = x_reshaped.shape[0],x_reshaped.shape[1]
        new_view = (sent_length * bs,x_reshaped.shape[2],x_reshaped.shape[3])        
        x_reshaped2 = x_reshaped.view(new_view) 
        # (max_sentence_length * bs ,e_char,max_word_length)
        
        x_conv = self.conv1d(x_reshaped2)  # (sent_length*bs,e_word,max_word_length-k+1).
        x_conv_out = F.relu(x_conv)
        x_conv_out = self.mp1d(x_conv_out).squeeze(-1) 
        # (sent_length*bs,e_word,1) to (sent_length*bs,e_word)
        
        x_conv_out = x_conv_out.view(sent_length,bs,self.e_word)
        
        return x_conv_out.contiguous()