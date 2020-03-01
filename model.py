import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = embedding
        #embedding will be of hidden_size
        self.GRU = nn.GRU(hidden_size, hidden_size, n_layers, dropout = (0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, inp_seq, inp_len, hidden=None):
        """
        1.Convert word indexes to embeddings.
        2.Pack padded batch of sequences for RNN module.
        3.Forward pass through GRU.
        4.Unpack padding.
        5.Sum bidirectional GRU outputs.
        6.Return output and final hidden state.
        """
        #(seq_len, batch)
        embed_seq = self.embedding(inp_seq) #(seq_len, batch, hidden)
        packed_inp = pack_padded_sequence(embed_seq, inp_len)
        gru_out, hidden = self.GRU(packed_inp, hidden)  #(seq_len, batch, 2*hidden)
        outputs, _ = pad_packed_sequence(gru_out)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:] #(seq_len, batch, hidden)
        return outputs, hidden

class Attn(nn.Module):
    """
    Luong Attention
    """
    def __init__(self, method, hidden_size):
        """
        1. Dot  2.General   3.Concat
        """
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not a valid method!")
        
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_outs):
        return torch.sum(hidden * encoder_outs, dim=2)

    def general_score(self, hidden, encoder_outs):
        energy = self.attn(encoder_outs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_outs):
        #for torch.Tensor.expand() - -1 will retain dimension
        energy = self.attn(torch.cat(hidden.expand(encoder_outs.size(0), -1, -1), encoder_outs, dim=2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outs):
        #encoder_outs (seq_len, batch, hidden)
        if self.method == "dot":
            energy = self.dot_score(hidden, encoder_outs)
        elif self.method == "general":
            energy = self.general_score(hidden, encoder_outs)
        else:
            energy = self.concat_score(hidden, encoder_outs)

        #(seq_len, batch)
        energy = energy.t() #(batch, seq_len)
        attn_energy = F.softmax(energy, dim=1).unsqueeze(1) #(batch, 1, seq_len)
        return attn_energy

class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_method, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.GRU = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.attn = Attn(attn_method, hidden_size)
        self.concat = nn.Linear(hidden_size*2, hidden_size) #Luong eq 5
        self.out = nn.Linear(hidden_size, output_size)  #Luong eq 6

    def forward(self, input_step, hidden, encoder_outs):
        """
        1.Get embedding of current input word.
        2.Forward through unidirectional GRU.
        3.Calculate attention weights from the current GRU output from (2).
        4.Multiply attention weights to encoder outputs to get new “weighted sum” context vector.
        5.Concatenate weighted context vector and GRU output using Luong eq. 5.
        6.Predict next word using Luong eq. 6 (without softmax).
        7.Return output and final hidden state.
        """
        #(1, batch)
        embed_inp = self.embedding(input_step)  #(1, batch, hidden) - embedding will be of hidden_size
        dropped_embed = self.embedding_dropout(embed_inp)
        gru_out, hidden = self.GRU(dropped_embed, hidden)   #(1, batch, hidden)
        attn_weights = self.attn(gru_out, encoder_outs) #(batch, 1, seq_len)
        #encoder_outs (seq_len, batch, hidden)
        weighted_sum = torch.bmm(attn_weights, encoder_outs.transpose(0,1)) #(batch, 1, hidden)

        concat_weighted_context = torch.cat((gru_out.squeeze(0), weighted_sum.squeeze(1)), dim=1)   #(batch, hidden*2)
        concat_out = torch.tanh(self.concat(concat_weighted_context))   #(batch, hidden)

        decoder_out = F.softmax(self.out(concat_out), dim=1)  #(batch, output_size)
        return decoder_out, hidden

if __name__ == "__main__":
    #sanity checks
    batch = 50
    hidden_size = 100
    attn_method = "concat"
    out_size = 200
    embed = nn.Embedding(batch, hidden_size)
    encoder = EncoderRNN(hidden_size, embed)
    print(encoder)
    decoder = LuongAttnDecoderRNN(attn_method, embed, hidden_size, out_size)
    print(decoder)