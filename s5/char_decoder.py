#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, 
        shape (length, batch_size, self.vocab_size)
        
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        inp_emb = self.decoderCharEmb(input) #(length,bs,char_emb_size)
        outp,new_dec_hidden = self.charDecoder(inp_emb,dec_hidden)
        # outp shape: (length,bs,hidden_size)
        scores = self.char_output_projection(outp) #(length,bs,vocab_size)
        return scores,new_dec_hidden
    
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size).
        "length" here is max_word_length, aka number of chars for the longest words ever in the batch
        "batch_size" is max_sent_length * bs
        Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        
        X = char_sequence[:-1] # c0 to cn for training
        s,dec_hidden = self.forward(X,dec_hidden)
        
        vocab_size = len(self.target_vocab.char2id)
        target = char_sequence[1:].view(-1).contiguous() # c1 to cn+1 for testing, flatten for loss
        scores = s.view(-1,vocab_size).contiguous()
        
        loss   = nn.CrossEntropyLoss(
            reduction= "sum", # When compute loss_char_dec, we take the sum, not average
            ignore_index=self.target_vocab.char_pad # not take into account pad character when compute loss
        )
        return loss(scores, target)
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. 
        ### Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} 
        ### (e.g., <START>,m,u,s,i,c,<END>). 
        ### Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss 
        ### and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        This is called only in inference and only when word-model produces an UNK
        If the translation contains any <UNK> tokens, 
        then for each of those positions, we use the word-based decoder’s combined
        output vector to initialize the CharDecoderLSTM’s initial h0 and c0, then use CharDecoderLSTM to
        generate a sequence of characters.

        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, 
        a tuple of two tensors of size (1, batch_size, hidden_size)
        
        
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, 
                            each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        bs = initialStates[0].shape[1]
        dec_hidden = initialStates
        inp = torch.LongTensor([[self.target_vocab.start_of_word]*bs], 
                           device=device) # shape (1,bs). 
        
        # Bptt is always 1 for inp (even for the loop below) instead of increment from 1
        # because we already update dec_hidden for LSTM
        # Otherwise, we will feed (1,bs) then (2,bs) ... (max_length-1,bs) into LSTM without
        # feeding dec_hidden. This wastes computation however.
        
        results=torch.LongTensor([[0]*bs],device=device) # (1,bs)
        for s in range(max_length):
            scores,dec_hidden = self.forward(inp,dec_hidden) #(1,bs,vocab_size)
            inp = F.softmax(scores,dim=2).argmax(dim=2) # (1,bs)
            # print(f'{s}: {inp}')
            # BIG FAT TODO: at s=0, even though inp is the same: 1,bs vector filled with start_of_word index
            # output vector (not sure at h or after blue-transformation) (1,bs) have different values. Pls do a LSTM check in notebook
            results = torch.cat([results,inp.detach()])

        decodedWords = []
        results = torch.transpose(results[1:],0,1).detach().cpu().numpy() # (bs,max_length)
        for row in results:
            row_str=[]
            for idx in row:
                if idx == self.target_vocab.end_of_word: break
                row_str.append(self.target_vocab.id2char[idx])
            decodedWords.append(''.join(row_str))


        return decodedWords

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char 
        ### to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. 
        ### That is, use the character '{' for <START> and '}' for <END>.

        ### END YOUR CODE