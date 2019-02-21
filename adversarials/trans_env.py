from adversarials.discriminator import TransDiscriminator
from adversarials.attacker import Attacker
from adversarials.adversarial_utils import load_translate_model, load_or_extract_near_vocab
from src.data.vocabulary import Vocabulary
from src.models import build_model
from src.optim import Optimizer
from src.optim.lr_scheduler import *
from src.utils.logging import *

import os
import torch
import torch.nn as nn
import numpy as np
import nltk.translate.bleu_score as bleu
import yaml

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD

def build_translate_model(victim_config,
                          victim_model_path,
                          vocab_src,
                          vocab_trg,
                          use_gpu,
                          ):
    """
    build translation env
    :param victim_config: victim configs
    :param victim_model_path: victim_models
    :param vocab_src: source vocabulary
    :param vocab_trg: target vocabulary
    :param use_gpu: gpu version of model
    :return: nmt_models used in the beam-search
    """
    translate_model_configs = victim_config["model_configs"]

    # build model for translation
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_trg.max_n_words,
                            **translate_model_configs)
    nmt_model.eval()
    params = load_translate_model(victim_model_path, map_location="cpu")
    nmt_model.load_state_dict(params)
    if use_gpu:
        nmt_model = nmt_model.cuda()
    return nmt_model

def load_embedding(model_D, model_path, device):
    pretrained_params = torch.load(model_path, map_location=device)
    model_D.src_embedding.embeddings.load_state_dict(
        {"weight": pretrained_params["encoder.embeddings.embeddings.weight"]},
        strict=True)
    model_D.trg_embedding.embeddings.load_state_dict(
        {"weight": pretrained_params["decoder.embeddings.embeddings.weight"]},
        strict=True)
    INFO("load embedding from NMT model")
    return

class Translate_Env(object):
    """
    wrap translate environment for multiple agents
    env needs parallel data to evaluate bleu_degredation
    :return: translation multiple sentences and return changed bleu
    """

    def __init__(self, attack_configs_path, save_to, use_gpu=True):
        with open(attack_configs_path.strip()) as f:
            configs=yaml.load(f)
        attack_configs = configs["attack_configs"]
        attacker_model_configs = configs["attacker_model_configs"]
        attacker_optim_configs = configs["attacker_optimizer_configs"]
        discriminator_model_configs = configs["discriminator_model_configs"]
        discriminator_optim_configs = configs["discriminator_optimizer_configs"]
        self.training_configs = configs["training_configs"]
        self.victim_config_path = attack_configs["victim_configs"]
        self.victim_model_path = attack_configs["victim_model"]
        self.use_gpu = use_gpu
        with open(self.victim_config_path.strip()) as v_f:
            print("open victim configs...%s" % self.victim_config_path)
            victim_configs = yaml.load(v_f)

        data_configs = victim_configs["data_configs"]
        self.src_vocab = Vocabulary(**data_configs["vocabularies"][0])
        self.trg_vocab = Vocabulary(**data_configs["vocabularies"][1])
        self.translate_model = build_translate_model(victim_configs,
                                                     self.victim_model_path,
                                                     vocab_src=self.src_vocab,
                                                     vocab_trg=self.trg_vocab,
                                                     use_gpu=self.use_gpu)
        self.w2p, self.w2vocab = load_or_extract_near_vocab(config_path=self.victim_config_path,
                                                  model_path=self.victim_model_path,
                                                  init_perturb_rate=attack_configs["init_perturb_rate"],
                                                  save_to=os.path.join(save_to, "near_vocab"),
                                                  save_to_full=os.path.join(save_to, "full_near_vocab"),
                                                  top_reserve=12)
        #########################################################
        # to update discriminator
        # discriminator_data_configs = attack_configs["discriminator_data_configs"]
        self.discriminator = TransDiscriminator(n_src_words=self.src_vocab.max_n_words,
                                                n_trg_words=self.trg_vocab.max_n_words,
                                                **discriminator_model_configs)
        if self.use_gpu:
            self.discriminator = self.discriminator.cuda()
            load_embedding(self.discriminator,
                           model_path=self.victim_model_path,
                           device="cuda")
        else:
            load_embedding(self.discriminator,
                           model_path=self.victim_model_path,
                           device="cpu")

        self.optim_D = Optimizer(name=discriminator_optim_configs["optimizer"],
                                 model=self.discriminator,
                                 lr=discriminator_optim_configs["learning_rate"],
                                 grad_clip=discriminator_optim_configs["grad_clip"],
                                 optim_args=discriminator_optim_configs["optimizer_params"])
        self.criterion_D = nn.CrossEntropyLoss()  # used in discriminator updates
        self.scheduler_D = None  # default as None
        if discriminator_optim_configs['schedule_method'] is not None:
            if discriminator_optim_configs['schedule_method'] == "loss":
                self.scheduler_D = ReduceOnPlateauScheduler(optimizer=self.optim_D,
                                                     **discriminator_optim_configs["scheduler_configs"]
                                                     )
            elif discriminator_optim_configs['schedule_method'] == "noam":
                self.scheduler_D = NoamScheduler(optimizer=self.optim_D, **discriminator_optim_configs['scheduler_configs'])
            elif discriminator_optim_configs["schedule_method"] == "rsqrt":
                self.scheduler_D = RsqrtScheduler(optimizer=self.optim_D, **discriminator_optim_configs["scheduler_configs"])
            else:
                WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(
                    discriminator_optim_configs['schedule_method']))
        ############################################################


    def prepare_D_data(self, attacker, seqs_x, seqs_y, use_gpu, batch_first=True):
        """
        using global_attacker to generate training data for discriminator
        :param attacker: prepare the data
        :param seqs_x: list of sources
        :param seqs_y: corresponding targets
        :param use_gpu: use_gpu
        :param batch_first: first dimension of seqs be batch
        :return: perturbed seqsx, seqsy, flags
        """
        def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):
            # pack seqs into tensor with pads
            batch_size = len(samples)
            sizes = [len(s) for s in samples]
            max_size = max(sizes)
            x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')
            for ii in range(batch_size):
                x_np[ii, :sizes[ii]] = samples[ii]
            if batch_first is False:
                x_np = np.transpose(x_np, [1, 0])
            x = torch.tensor(x_np)
            if cuda is True:
                x = x.cuda()
            return x
        seqs_x = list(map(lambda s: [BOS] + s + [EOS], seqs_x))
        x = _np_pad_batch_2D(samples=seqs_x, pad=PAD,
                             cuda=use_gpu, batch_first=batch_first)
        # training mode attack: randomly choose half of the seqs to attack
        attacker.eval()
        seqs_x, flags = attacker.seq_attack(seqs_x,
                                            self.w2p, self.w2vocab,
                                            training_mode=True)
        seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
        y = _np_pad_batch_2D(seqs_y, pad=PAD,
                             cuda=use_gpu, batch_first=batch_first)
        if use_gpu:
            flags = flags.cuda()
        return x, y, flags

    def compute_D_forward(self, model, criterion,
                          seqs_x, seqs_y, gold_flags,
                          evaluate=False):
        """
        get loss according to criterion
        :return: loss value
        """
        if not evaluate:
            # set components to training mode
            model.train()
            criterion.train()
            with torch.enable_grad():
                class_probs = model(seqs_x, seqs_y)
                loss = criterion(class_probs, gold_flags)
            torch.autograd.backward(loss)
            return loss.item()
        else:
            # set components to evaluation mode
            model.eval()
            criterion.eval()
            with torch.no_grad():
                class_probs = model(seqs_x, seqs_y)
                loss = criterion(class_probs, gold_flags)
        return loss.item()

    def update_discriminator(self,
                             data_iterator,
                             attacker_model,
                             minimum_update_steps=10,
                             accuracy_bound=None):
        """
        update discriminator
        :param data_iterator: provide batched labels for training
        :param minimum_update_steps: minimum update steps,
                    also the discriminator evaluate steps
        :param accuracy_bound: update until accuracy reaches the bound
        :return:
        """
        INFO("update discriminator")
        self.optim_D.zero_grad()
        step = 0
        while True:
            for batch in data_iterator:
                step += 1
                if self.scheduler_D is not None:
                    # override learning rate in self.optim_D
                    self.scheduler_D.step(global_step=step)
                seqs_x, seqs_y = batch  # returned tensor type of the data
                try:
                    x, y, flags = self.prepare_D_data(attacker_model,
                                                      seqs_x, seqs_y,
                                                      use_gpu=self.use_gpu)
                    loss = self.compute_D_forward(self.discriminator, criterion=self.criterion_D,
                                                  seqs_x=x,
                                                  seqs_y=y, gold_flags=flags)
                    self.optim_D.step()
                    print("discriminator loss:", loss)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory, skipping batch")
                        self.optim_D.zero_grad()
                    else:
                        raise e
                # valid for accuracy / check for break (if any)
                if accuracy_bound and step % minimum_update_steps == 0:
                    pass


    def perturb(self, batch_sent, index, actions, train_mode):
        for src, trg in batch_sent:
            pass


    def step(self, batch_sent, index, actions):
        """
        run env and get bleu loss
        :param batch_sent: batched parallel sentences
        :param index: index (position) to perturb
        :param actions: whether to perturb (boolean vector in shape [batch, action_space] )
        :return: rewards from the environments
        """
        discriminator_reward = 0.1
        # action results
        batch_sent = perturb(batch_sent, index, actions)

        # check for discriminator：
        self.discriminator()
