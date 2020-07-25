import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torchvision
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# image shape (C, H, W)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 2
z_dim = 100
input_text = torch.randn([batch_size ,1024]).to(device) # s_ca input
noise = torch.randn(batch_size, z_dim).to(device) # generator input



class Text_Encoder(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_dim,
                  n_layers,
                   dropout_p,
                    max_length,
                     rnn_type):
        super(Text_Encoder, self).__init__()

        self.rnn_type = rnn_type
        self.vocap_size = vocab_size 
        self.embedding_dim = embedding_dim #embedding size
        self.drop_rate = 0.5 #dropout rate
        self.hidden_dim = hidden_dim # word dim
        self.num_layers = n_layers
        self.bidirectional = True # bidirectional option

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.hidden_dim = self.hidden_dim // self.num_directions
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocap_size, embedding_dim=self.embedding_dim)
        self.dropout = nn.Dropout(self.drop_rate)

        # Added for LM
        self.out = nn.Linear(self.embedding_dim, self.vocap_size, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=2)

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size = self.embedding_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers,
                               batch_first=True,
                               dropout=self.drop_rate,
                               bidirectional=self.bidirectional)
        else:
            self.rnn = nn.GRU(input_size = self.embedding_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=self.drop_rate,
                              bidirectional=self.bidirectional)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        print(weight)
        if self.rnn_type == 'LSTM':
            return (weight.new(self.num_layers * self.num_directions, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.num_layers * self.num_directions, batch_size, self.hidden_dim).zero_())
        else:
            return weight.new(self.num_layers * self.num_directions, batch_size, self.hidden_dim).zero_()

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input : [B, N_steps]
        print(captions)
        embed = self.embedding_layer(captions)
        embed = self.dropout(embed)
        # embed : [B, N_steps, Embeding_dim]

        cap_lens = cap_lens.data.tolist()
        embed = pack_padded_sequence(embed, cap_lens, batch_first=True)
        # embed : [B, N_steps, Embedding_dim] # Sorting & Removing the Paddings

        
        out, hidden = self.rnn(embed, hidden)
        # out : [B, N_steps, N_directions * N_hidden]
        # hidden : [N_layers * N_directions, B, N_hidden]

        out = pad_packed_sequence(out, batch_first=True)[0]
        # out : [B, N_steps, N_directions * N_hidden] # Unsorting? & Padding
        
        out = self.out(out)

        y_hat = self.log_softmax(x)
               
        return y_hat

        # words_emb = out.transpose(1, 2)
        # # words_emb : [B, N_directions * N_hidden, N_steps ]


        # if self.rnn_type == 'LSTM':
        #     sentence_emb = hidden[0].transpose(0, 1).contiguous()
        # else:
        #     sentence_emb = hidden.transpose(0, 1).contiguous()
        # sentence_emb = sentence_emb.view(-1, self.hidden_dim * self.num_directions)

        # return words_emb, sentence_emb
