import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.embeddings import Embeddings
from src.modules.rnn import RNN
from src.data.vocabulary import PAD


class TransDiscriminator(nn.Module):
    """
    discriminate whether the trg (y) is a translation of a src (x)
    """
    def __init__(self,
                 n_src_words,
                 n_trg_words,
                 d_word_vec,
                 d_model,
                 dropout=0.0,
                 **kwargs,
                 ):
        super(TransDiscriminator, self).__init__()
        # the embedding is pre-trained and without dropout layer
        self.src_embedding = Embeddings(num_embeddings=n_src_words,
                                        embedding_dim=d_word_vec,
                                        dropout=0.0,
                                        add_position_embedding=False)
        self.trg_embedding = Embeddings(num_embeddings=n_trg_words,
                                        embedding_dim=d_word_vec,
                                        dropout=0.0,
                                        add_position_embedding=False)
        if not kwargs["update_embedding"]:
            for param in self.src_embedding.parameters():
                param.requires_grad = False
            for param in self.trg_embedding.parameters():
                param.requires_grad = False

        self.src_gru = RNN(type="gru", batch_first=True, input_size=d_word_vec,
                           hidden_size=d_model, bidirectional=True)
        self.trg_gru = RNN(type="gru", batch_first=True, input_size=d_word_vec,
                           hidden_size=d_model, bidirectional=True)
        # whether the (x,y) is a translation pair
        self.ffn = nn.Linear(in_features=4*d_model, out_features=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        given src and trg, output classification label
        :param x: batched src in shape [batch_size, max_seq_len]
        :param y: batched trg in shape [batch_size, max_seq_len]
        :return: labels indicating probability in shape [batch_size, 2]
        """
        x_mask = x.detach().eq(PAD)
        y_mask = y.detach().eq(PAD)
        x_emb = self.src_embedding(x)
        y_emb = self.trg_embedding(y)

        ctx_x, _ = self.src_gru(x_emb, x_mask)
        ctx_y, _ = self.trg_gru(y_emb, y_mask)

        x_pad_mask = 1.0 - x_mask.float()
        y_pad_mask = 1.0 - y_mask.float()
        x_ctx_mean = (ctx_x * x_pad_mask.unsqueeze(2)).sum(1) / x_pad_mask.unsqueeze(2).sum(1)
        y_ctx_mean = (ctx_y * y_pad_mask.unsqueeze(2)).sum(1) / y_pad_mask.unsqueeze(2).sum(1)
        output = self.ffn(self.dropout(torch.cat((x_ctx_mean, y_ctx_mean), dim=-1)))
        output = F.softmax(output, dim=-1)
        return output

    def encode_src(self, x):
        """
        encode batched x into representation for attacker.
        :param x: batched x [batch_size, max_seq_len]
        :return: avg GRU_hiddens representing seq_x in shape [batch_size, dim]
        """
        x_mask = x.detach().eq(PAD)
        x_emb = self.src_embedding(x)

        ctx_x, _ = self.src_gru(x_emb, x_mask)
        x_pad_mask = 1.0 - x_mask.float()
        x_ctx_mean = (ctx_x * x_pad_mask.unsqueeze(2)).sum(1) / x_pad_mask.unsqueeze(2).sum(1)
        return x_ctx_mean

    def reset(self):
        def weights_init(tensor):
            if tensor.ndimension() == 1:
                nn.init.constant_(tensor, val=0.0)
            else:
                nn.init.xavier_normal_(tensor)
            return tensor

        # collect module parameters
        for name, p in self.named_parameters():
            # print(name, " reset")
            weights_init(p)
        return
