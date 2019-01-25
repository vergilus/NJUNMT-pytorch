import torch
import torch.nn as nn
import torch.nn.functional as F
import src.utils.init as my_init
from src.modules.embeddings import Embeddings
from src.modules.rnn import RNN
from src.data.vocabulary import PAD


class Attacker(nn.Module):
    """
    which is a LSTM (GRU), the output vector is then projected to actor actions and critic value
    using actor critic.

    states: current hidden representation of the sentence
    actions: perturb or not (action_space=2)
    rewards: BLEU values and discriminator values
    new_states: hidden representation of the perturbed sentence

    """
    def __init__(self,
                 n_words,
                 action_space=2,
                 input_size=512,
                 hidden_size=256,
                 bridge_type="zero",
                 dropout_rate=0.0):

        super(Attacker, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = Embeddings(num_embeddings=n_words,
                                    embedding_dim=input_size,
                                    dropout=0.0,
                                    add_position_embedding=False)
        self.gru = RNN(type="gru", batch_first=True,
                       input_size=input_size, hidden_size=hidden_size)
        # inputs: current input, previous hidden and previous avg_seqs
        self.ctx_linear = nn.Linear(in_features=2*hidden_size,
                                    out_features=hidden_size)
        self.hidden_linear = nn.Linear(in_features=hidden_size,
                                       out_features=hidden_size)
        self.input_linear = nn.Linear(in_features=input_size,
                                      out_features=hidden_size)
        # outputs: actor distribution and critic value
        self.actor_linear = nn.Linear(in_features=hidden_size,
                                      out_features=action_space)
        self.critic_linear = nn.Linear(in_features=hidden_size,
                                       out_features=1)
        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

    def _reset_parameters(self):
        my_init.default_init(self.ctx_linear.weight)
        my_init.default_init(self.input_linear.weight)
        my_init.default_init(self.hidden_linear.weight)
        my_init.default_init(self.actor_linear.weight)
        my_init.default_init(self.critic_linear.weight)

    def init_RNN(self, ctx, mask):
        # generate avg ctx_src vec and initial hidden states for RNN
        if self.bridge_type =="zero":
            pass
        return

    def forward(self, x, sent_state, one_step=False, cache=None):
        """
        for sentence x, compute actor actions and critic value
        :param x: batched sentences [batch, sent_len]
        :return:
        """
        x_mask = x.detach().eq(PAD)  # PAD mask
        x_emb = self.embedding(x)  # [batch, sent_len, dim]
        if one_step:
            pass
        else:
            pass


