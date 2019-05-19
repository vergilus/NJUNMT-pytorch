from adversarials.discriminator import TransDiscriminator
from adversarials.adversarial_utils import load_translate_model, load_or_extract_near_vocab
from src.data.vocabulary import Vocabulary
from src.models import build_model
from src.optim import Optimizer
from src.optim.lr_scheduler import *
from src.utils.logging import *
from src.utils.common_utils import *
from src.decoding import beam_search, ensemble_beam_search

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
                          device,
                          ):
    """
    build translation env
    :param victim_config: victim configs
    :param victim_model_path: victim_models
    :param vocab_src: source vocabulary
    :param vocab_trg: target vocabulary
    :param device: gpu version of model
    :return: nmt_models used in the beam-search
    """
    translate_model_configs = victim_config["model_configs"]

    # build model for translation
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_trg.max_n_words,
                            **translate_model_configs)
    nmt_model.eval()
    INFO("load embedding params to device %s" % device)
    params = load_translate_model(victim_model_path, map_location=device)
    nmt_model.load_state_dict(params)
    nmt_model.to(device)
    INFO("finished building translation model for environment on %s" % device)
    return nmt_model


def load_embedding(model_D, model_path, device):
    pretrained_params = torch.load(model_path, map_location=device)
    model_D.src_embedding.embeddings.load_state_dict(
        {"weight": pretrained_params["encoder.embeddings.embeddings.weight"]},
        strict=True)
    model_D.trg_embedding.embeddings.load_state_dict(
        {"weight": pretrained_params["decoder.embeddings.embeddings.weight"]},
        strict=True)
    INFO("load embedding from NMT model to device %s" % device)
    return


class Translate_Env(object):
    """
    wrap translate environment for multiple agents
    env needs parallel data to evaluate bleu_degredation
    state of the env is defined as the batched src labels and current target index
    environment yields rewards based on discriminator and finally by sentence-level BLEU
    :return: translation multiple sentences and return changed bleu
    """
    def __init__(self, attack_configs,
                 discriminator_configs,
                 src_vocab, trg_vocab,
                 data_iterator,
                 save_to,
                 device="cpu",
                 ):
        """
        initiate translation environments, needs a discriminator and translator
        :param attack_configs: attack configures dictionary
        :param save_to: discriminator models
        :param data_iterator: use to provide data for environment initiate
        the directory of the src sentences
        :param device: (string) devices to allocate variables("cpu", "cuda:*")
        default as cpu
        """
        self.data_iterator = data_iterator
        discriminator_model_configs = discriminator_configs["discriminator_model_configs"]
        discriminator_optim_configs = discriminator_configs["discriminator_optimizer_configs"]
        self.victim_config_path = attack_configs["victim_configs"]
        self.victim_model_path = attack_configs["victim_model"]
        # determine devices
        self.device = device
        with open(self.victim_config_path.strip()) as v_f:
            print("open victim configs...%s" % self.victim_config_path)
            victim_configs = yaml.load(v_f)

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.translate_model = build_translate_model(victim_configs,
                                                     self.victim_model_path,
                                                     vocab_src=self.src_vocab,
                                                     vocab_trg=self.trg_vocab,
                                                     device=self.device)
        self.translate_model.eval()
        self.w2p, self.w2vocab = load_or_extract_near_vocab(config_path=self.victim_config_path,
                                                            model_path=self.victim_model_path,
                                                            init_perturb_rate=attack_configs["init_perturb_rate"],
                                                            save_to=os.path.join(save_to, "near_vocab"),
                                                            save_to_full=os.path.join(save_to, "full_near_vocab"),
                                                            top_reserve=12,
                                                            emit_as_id=True)
        #########################################################
        # to update discriminator
        # discriminator_data_configs = attack_configs["discriminator_data_configs"]
        self.discriminator = TransDiscriminator(n_src_words=self.src_vocab.max_n_words,
                                                n_trg_words=self.trg_vocab.max_n_words,
                                                **discriminator_model_configs)
        self.discriminator.to(self.device)

        load_embedding(self.discriminator,
                       model_path=self.victim_model_path,
                       device=self.device)

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
                                                            **discriminator_optim_configs["scheduler_configs"])
            elif discriminator_optim_configs['schedule_method'] == "noam":
                self.scheduler_D = NoamScheduler(optimizer=self.optim_D,
                                                 **discriminator_optim_configs['scheduler_configs'])
            elif discriminator_optim_configs["schedule_method"] == "rsqrt":
                self.scheduler_D = RsqrtScheduler(optimizer=self.optim_D,
                                                  **discriminator_optim_configs["scheduler_configs"])
            else:
                WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(
                    discriminator_optim_configs['schedule_method']))
        ############################################################
        self._init_state()

    def _init_state(self):
        """
        initiate batched sentences / origin_bleu / index (start from first label, no BOS/EOS)
        the initial state of the environment
        :return: env states (the src, index)
        """
        self.index = 1
        self.origin_bleu = []
        batch = next(self.data_iterator)
        assert len(batch) == 3, "must be provided with line index (check for data_iterator)"
        # training, parallel trg is provided
        _, seqs_x, self.seqs_y = batch
        self.sent_len = [len(x) for x in seqs_x] # for terminal signals
        self.terminal_signal = [0] * len(seqs_x)  # for terminal signals

        self.padded_src, self.padded_trg = self.prepare_data(seqs_x=seqs_x,
                                                             seqs_y=self.seqs_y,
                                                             cuda=True if self.device!="cpu" else False)
        self.origin_result = self.translate()
        # calculate BLEU scores for the top candidate
        for index, sent_t in enumerate(self.seqs_y):
            bleu_t = bleu.sentence_bleu(references=[sent_t], hypothesis=self.origin_result[index])
            self.origin_bleu.append(bleu_t)
        return self.padded_src.cpu().numpy()

    def get_src_vocab(self):
        return self.src_vocab

    def reset(self):
        return self._init_state()

    def reset_data_iter(self, data_iter):  # reset data iterator with provided iterator
        self.data_iterator = data_iter
        return

    def reset_discriminator(self):
        self.discriminator.reset()
        load_embedding(self.discriminator,
                       model_path=self.victim_model_path,
                       device=self.device)

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
        x, flags = attacker.seq_attack(x, self.w2vocab,
                                       training_mode=True)

        seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))

        y = _np_pad_batch_2D(seqs_y, pad=PAD,
                             cuda=use_gpu, batch_first=batch_first)
        if use_gpu:
            flags = flags.cuda()

        # # print trace
        # flag_list = flags.cpu().numpy().tolist()
        # x_list = x.cpu().numpy().tolist()
        # for i in range(len(flag_list)):
        #     if flag_list[i]==1:
        #         print(self.src_vocab.ids2sent(seqs_x[i]))
        #         print(self.src_vocab.ids2sent(x_list[i]))
        #         print(self.trg_vocab.ids2sent(seqs_y[i]))
        return x, y, flags

    def prepare_data(self, seqs_x, seqs_y=None, cuda=False, batch_first=True):
        """
        Args:
            eval ('bool'): indicator for eval/infer.
        Returns: padded data matrices
        """
        def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):
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
                             cuda=cuda, batch_first=batch_first)
        if seqs_y is None:
            return x
        seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
        y = _np_pad_batch_2D(seqs_y, pad=PAD,
                             cuda=cuda, batch_first=batch_first)

        return x, y

    def acc_validation(self, attacker, use_gpu):
        self.discriminator.eval()
        acc = 0
        sample_count = 0
        for i in range(5):
            try:
                batch = next(self.data_iterator)
            except StopIteration:
                batch = next(self.data_iterator)
            seq_nums, seqs_x, seqs_y = batch
            x, y, flags = self.prepare_D_data(attacker,
                                              seqs_x, seqs_y, use_gpu)
            # set components to evaluation mode
            self.discriminator.eval()
            with torch.no_grad():
                preds = self.discriminator(x, y).argmax(dim=-1)
                acc += torch.eq(preds, flags).sum()
                sample_count += preds.size(0)
        acc = acc.float() / sample_count
        return acc.item()

    def compute_D_forward(self, seqs_x, seqs_y, gold_flags,
                          evaluate=False):
        """
        get loss according to criterion
        :param: gold_flags=1 if perturbed, otherwise 0
        :return: loss value
        """
        if not evaluate:
            # set components to training mode(dropout layers)
            self.discriminator.train()
            self.criterion_D.train()
            with torch.enable_grad():
                class_probs = self.discriminator(seqs_x, seqs_y)
                loss = self.criterion_D(class_probs, gold_flags)
            torch.autograd.backward(loss)
            return loss.item()
        else:
            # set components to evaluation mode(dropout layers)
            self.discriminator.eval()
            self.criterion_D.eval()
            with torch.no_grad():
                class_probs = self.discriminator(seqs_x, seqs_y)
                loss = self.criterion_D(class_probs, gold_flags)
        return loss.item()

    def update_discriminator(self,
                             data_iterator,
                             attacker_model,
                             base_steps=0,
                             min_update_steps=20,
                             max_update_steps=300,
                             accuracy_bound=0.8,
                             summary_writer=None):
        """
        update discriminator
        :param data_iterator: (data_iterator type) provide batched labels for training
        :param attacker_model: attacker to generate training data for discriminator
        :param base_steps: used for saving
        :param min_update_steps: (integer) minimum update steps,
                    also the discriminator evaluate steps
        :param max_update_steps: (integer) maximum update steps
        :param accuracy_bound: (float) update until accuracy reaches the bound
                    (or max_update_steps)
        :param summary_writer: used to log discriminator learning information
        :return: steps and test accuracy as trust region
        """
        INFO("update discriminator")
        self.optim_D.zero_grad()
        attacker_model = attacker_model.to(self.device)
        step = 0
        while True:
            try:
                batch = next(self.data_iterator)
            except StopIteration:
                batch = next(self.data_iterator)
            # update the discriminator
            step += 1
            if self.scheduler_D is not None:
                # override learning rate in self.optim_D
                self.scheduler_D.step(global_step=step)
            _, seqs_x, seqs_y = batch  # returned tensor type of the data
            try:
                x, y, flags = self.prepare_D_data(attacker_model,
                                                  seqs_x, seqs_y,
                                                  use_gpu=True if self.device!="cpu" else False)
                loss = self.compute_D_forward(seqs_x=x,
                                              seqs_y=y,
                                              gold_flags=flags)
                self.optim_D.step()
                print("discriminator loss:", loss)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory, skipping batch")
                    self.optim_D.zero_grad()
                else:
                    raise e

            # valid for accuracy / check for break (if any)
            if step % min_update_steps == 0:
                acc = self.acc_validation(attacker_model, use_gpu=True if self.device != "cpu" else False)
                print("discriminator acc: %2f" % acc)
                summary_writer.add_scalar("discriminator", scalar_value=acc, global_step=base_steps+step)
                if accuracy_bound and acc > accuracy_bound:
                    INFO("discriminator reached training acc bound, updated.")
                    return base_steps+step, acc

            if step > max_update_steps:
                acc = self.acc_validation(attacker_model, use_gpu=True if self.device != "cpu" else False)
                print("discriminator acc: %2f" % acc)
                INFO("Reach maximum discriminator update. Finished.")
                return base_steps+step, acc   # stop updates


    def translate(self, inputs=None):
        """
        translate the self.perturbed_src
        :param input: if None, translate perturbed sequences stored in the environments
        :return: list of translation results
        """
        if inputs is None:
            inputs = self.padded_src
        with torch.no_grad():
            perturbed_results = beam_search(self.translate_model, beam_size=5, max_steps=150,
                                            src_seqs=inputs, alpha=-1.0,
                                            )
        perturbed_results = perturbed_results.cpu().numpy().tolist()
        # only use the top result from the result
        result = []
        for sent in perturbed_results:
            sent = [wid for wid in sent[0] if wid != PAD]
            result.append(sent)

        return result

    def step(self, actions):
        """
        step update for the environment: finally update self.index
        this is defined as inference of the environments
        :param actions: whether to perturb (action distribution vector
                    in shape [batch, 1])on current index
                 *  result of torch.argmax(actor_output_distribution, dim=-1)
                    test: actions = actor_output_distribution.argmax(dim=-1)
                    or train: actions = actor.output_distribution.multinomial(dim=-1)
                    can be on cpu or cuda.
        :return: updated states/ rewards/ terminal signal from the environments
                 reward (float), terminal_signal (boolean)
        """
        with torch.no_grad():
            terminal = False  # default is not terminated
            batch_size = actions.shape[0]
            reward = 0
            inputs = self.padded_src[:, self.index]
            inputs_mask = 1-inputs.eq(PAD)
            target_of_step = []
            # modification on sequences (state)
            for batch_index in range(batch_size):
                word_id = inputs[batch_index]
                # print(word_id.item())
                # print(word_id.item(),len(w2vocab[word_id.item()]))
                target_word_id = self.w2vocab[word_id.item()][np.random.choice(len(self.w2vocab[word_id.item()]), 1)[0]]
                target_of_step += [target_word_id]
            if self.device != "cpu" and not actions.is_cuda:
                actions = actions.cuda()
                actions *= inputs_mask  # PAD is neglect
            # override the state src with random choice from candidates
            self.padded_src[:, self.index] *= (1 - actions)
            adjustification_ = torch.tensor(target_of_step)
            if self.device != "cpu":
                adjustification_ = adjustification_.cuda()
            self.padded_src[:, self.index] += adjustification_ * actions

            # update sequences' pointer
            self.index += 1

            """ run discriminator check for terminal signals, update local terminal list
            True: all sentences in the batch is defined as false by self.discriminator
            False: otherwise
            """
            # get discriminator distribution on the current src state
            discriminate_out = self.discriminator(self.padded_src, self.padded_trg)
            self.terminal_signal = self.terminal_signal or discriminate_out.detach().argmax(dim=-1).cpu().numpy().tolist()
            signal = (1 - discriminate_out.argmax(dim=-1)).sum().item()
            if signal == 0 or self.index == self.padded_src.shape[1]-1:
                terminal = True  # no need to further explore or reached EOS for all src

            """ collect rewards on the current state
            """
            # calculate intermediate survival rewards
            if not terminal:  # survival rewards for survived objects
                distribution, discriminate_index = discriminate_out.max(dim=-1)
                distribution = distribution.detach().cpu().numpy()
                discriminate_index = (1 - discriminate_index).cpu().numpy()
                survival_value = distribution * discriminate_index * (1-np.array(self.terminal_signal))
                reward += survival_value.sum()/2
            else:  # penalty for intermediate termination
                reward = -1 * batch_size

            # only check for finished BLEU degradation when determined on the last label
            if self.index == self.padded_src.shape[1]-1:
                # translate calculate padded_src:
                perturbed_result = self.translate()
                # calculate final BLEU degredation:
                bleu_degrade = []
                for i, sent in enumerate(self.seqs_y):
                    # sentence is still surviving
                    if self.index >= self.sent_len[i]-1 and self.terminal_signal[i] == 0:
                        degraded_value = self.origin_bleu[i]-bleu.sentence_bleu(references=[sent],hypothesis=perturbed_result[i])
                        bleu_degrade.append(degraded_value)
                    else:
                        bleu_degrade.append(0.0)
                reward += sum(bleu_degrade) * 10

            reward = reward/batch_size

        return self.padded_src.cpu().numpy(), reward, terminal,

