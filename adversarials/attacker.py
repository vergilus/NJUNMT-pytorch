import torch
import torch.nn as nn
import torch.nn.functional as F
import src.utils.init as my_init
from src.utils.common_utils import *
from src.modules.embeddings import Embeddings
from src.modules.rnn import RNN
from src.data.vocabulary import PAD
import numpy as np


class Attacker(nn.Module):
    """
    which is MLP, the output vector is then projected to actor actions and critic value
    using actor critic.

    states: current hidden representation of the sentence
    actions: perturb or not (action_space=2)
    rewards: BLEU values and discriminator values
    new_states: hidden representation of the perturbed sentence

    """
    def __init__(self,
                 n_words,
                 action_space=2,
                 action_roll_steps=1,
                 d_word_vec=512,
                 d_model=256,
                 dropout=0.0,
                 **kwargs):
        super(Attacker, self).__init__()
        self.action_roll_steps = action_roll_steps
        self.action_space = action_space
        self.input_size = d_word_vec
        self.hidden_size = d_model
        self.src_embedding = Embeddings(num_embeddings=n_words,
                                        embedding_dim=self.input_size,
                                        dropout=0.0,
                                        add_position_embedding=False)
        # label representation
        self.src_gru = RNN(type="gru", batch_first=True, input_size=self.input_size,
                           hidden_size=self.hidden_size, bidirectional=True)

        # MLP attacker
        self.attacker = nn.Linear(in_features=self.hidden_size,
                                  out_features=self.hidden_size)

        # inputs: current input, avg_seqs as ctx
        self.ctx_linear = nn.Linear(in_features=2*self.hidden_size,
                                    out_features=self.hidden_size)
        self.input_linear = nn.Linear(in_features=self.input_size,
                                      out_features=self.hidden_size)

        # outputs: actor distribution and critic value
        self.attacker_linear = nn.Linear(in_features=self.hidden_size,
                                         out_features=self.action_space)
        self.critic_linear = nn.Linear(in_features=self.hidden_size,
                                       out_features=1)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        my_init.default_init(self.attacker.weight)
        my_init.default_init(self.ctx_linear.weight)
        my_init.default_init(self.input_linear.weight)

        my_init.default_init(self.attacker_linear.weight)
        my_init.default_init(self.critic_linear.weight)

    def preprocess(self, x, label):
        """
        for sentence x, compute actor actions and critic value
        :param label: batched labels [batch, 3], including previous\current\next label
        :param x: batched sentences [batch, max_seq_len]
        :return: batched actions in shape [batch, 2] and value estimation for the state [batch, 1]
        """
        label_emb = self.src_embedding(label)  # [batch, 3, dim]

        label_emb = label_emb.sum(dim=1)  # [batch, dim]
        x_emb = self.src_embedding(x)  # [batch, max_seq_len, dim]
        x_mask = x.detach().eq(PAD)
        ctx_x, _ = self.src_gru(x_emb, x_mask)
        x_pad_mask = 1.0 - x_mask.float()
        x_ctx_mean = (ctx_x * x_pad_mask.unsqueeze(2)).sum(1) / x_pad_mask.unsqueeze(2).sum(1)

        attack_feature = self.input_linear(label_emb)+self.ctx_linear(x_ctx_mean)
        return attack_feature

    def forward(self, x, label):
        """
        for sentence x, compute actor actions and critic value
        :param label: batched labels [batch, 1]
        :param x: batched sentences [batch, max_seq_len]
        :return: batched actions in shape [batch, 2] and
                  value estimation for the state averaged value over batch
        """
        attack_feature = self.preprocess(x, label)
        attack_out = F.softmax(self.dropout(self.attacker_linear(attack_feature)), dim=-1)
        critic_out = F.relu(self.dropout(self.critic_linear(attack_feature)))
        critic_out = critic_out.mean()
        return attack_out, critic_out

    def get_attack(self, x, label):
        """
        launch and return the attacked output vec (distribution on the actionspace)
        :param x: padded matrix of sequences labels
        :param label: current labels to be attacked
        :return: distribution on the action-space
        """
        # returns the attack actions
        attack_feature = self.preprocess(x, label)
        attack_out = F.softmax(self.dropout(self.attacker_linear(attack_feature)), dim=-1)
        return attack_out

    def get_critic(self, x, label):
        """
        value function estimation on the state (x)
        :param x: padded matrix of sequences labels
        :param label: current labels to be attacked
        :return: distribution on the action-space
        """
        # return value function of the current state
        attack_feature = self.preprocess(x, label)
        critic_out = F.relu(self.dropout(self.critic_linear(attack_feature)))
        critic_out = critic_out.mean()
        return critic_out

    def seq_attack(self, seqs_x_ids, w2p, w2vocab, training_mode):
        """ launch the attack for batch of sequences until EOS
        this is for data-preparation of training discriminator
        used in inference and prepare training data for discriminator
        :param seqs_x_ids: padded tensor variable of batched labels [batch, max_seq_len]
        :param w2p: dictionary [word: probability] (must be saved as ids)
        :param w2vocab: dictionary [word: near candidates] (must be saved as ids)
        :param training_mode: for discriminator training, randomly choose half of the x to perturb
        :return: torch variable perturbed seqs_x and flags
            perturbed: flag = 1, otherwise 0
        """
        perturbed_x_ids = seqs_x_ids.clone()
        with torch.no_grad():
            batch_size, max_steps = seqs_x_ids.shape
            for t in range(1, max_steps-1):  # ignore BOS and EOS
                inputs = seqs_x_ids[:, t-1:t+1]
                actions = self.get_attack(x=perturbed_x_ids, label=inputs).argmax(dim=-1)
                # inputs_np = inputs.numpy()
                target_of_step = []
                for batch_index in range(batch_size):
                    word_id = inputs[batch_index][1]
                    # print(word_id.item())
                    # print(word_id.item(),len(w2vocab[word_id.item()]))
                    target_word_id = w2vocab[word_id.item()][np.random.choice(len(w2vocab[word_id.item()]), 1)[0]]
                    target_of_step += [target_word_id]
                # override the perturbed results with random choice from candidates
                perturbed_x_ids[:, t] *= (1-actions)
                adjustification_ = torch.tensor(target_of_step, device=inputs.device)
                if GlobalNames.USE_GPU:
                    adjustification_ = adjustification_.cuda()
                perturbed_x_ids[:, t] += adjustification_ * actions
            # by default we perturb all the seqs_x
            flags = torch.ones(batch_size, device=inputs.device)
            if training_mode:
                # randomly choose half of the sentences
                flags = torch.bernoulli(0.5*flags).long()
                perturbed_x_ids *= flags.unsqueeze(dim=-1)
                perturbed_x_ids += seqs_x_ids*(1-flags).unsqueeze(dim=-1)

        return perturbed_x_ids, flags

    def sync_from(self, attack_model):
        # synchronize model parameteres from other attacker
        self.load_state_dict(attack_model.state_dict())

    def ensure_shared_grads(self, shared_model):
        """
        for multi-thread(agent) learning updates

        :param shared_model: a global model to share grad parameters
        :return:
        """
        for param, shared_param in zip(self.parameters(),
                                       shared_model.parameters()):
            # only copy the gradients to the shared when shared grad is None
            # the sync in the training process means that the update is processed
            # thread by thread
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
