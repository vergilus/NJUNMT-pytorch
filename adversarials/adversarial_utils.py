#!/usr/bin/env python
#coding=UTF-8

import os
import torch
import yaml
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from src.data.vocabulary import Vocabulary
from src.data.vocabulary import PAD, BOS, EOS, UNK
import subprocess
import time

models_path = ["/home/public_data/nmtdata/nmt-baselines/transformer-wmt15-enfr/small_baseline/baseline_enfr_save/transformer.best.final",
               "/home/zouw/NJUNMT-pytorch/scripts/save_zhen_bpe/transformer.best.final"]
configs_path = ["/home/public_data/nmtdata/nmt-baselines/transformer-wmt15-enfr/small_baseline/transformer_wmt15_en2fr.yaml",
                "/home/zouw/NJUNMT-pytorch/configs/transformer_nist_zh2en_bpe.yaml"]


def random_text_selection(config_path,
                          data_size=100,
                          save_log="random_sents"):
    # load configs
    np.random.seed(32767)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    data_configs = configs["data_configs"]
    return_set = []
    with open(data_configs["train_data"][0], "r") as src, open(save_log, "w") as out:
        i = 0
        for line in src:
            if np.random.uniform() > 0.5 and i < data_size:
                i += 1
                out.write(line)
                return_set += [line]
    return return_set


def random_pair_selection(config_path,
                          data_size=100,
                          save_log="random_sents"):
    """
    randomly choose from parallel data, and save to the save_logs
    :param config_path:
    :param data_size:
    :param save_log:
    :return: random selected pairs
    """
    np.random.seed(32767)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    data_configs = configs["data_configs"]
    with open(data_configs["train_data"][0], "r") as src, \
        open(data_configs["train_data"][1], "r") as trg, \
        open(save_log+".src", "w") as out_src, open(save_log+".trg", "w") as out_trg:
        counter=0
        return_src=[]
        return_trg=[]
        for sent_s, sent_t in zip(src,trg):
            if np.random.uniform()<0.2 and counter<data_size:
                counter += 1
                out_src.write(sent_s)
                out_trg.write(sent_t)
                return_src+=[sent_s.strip()]
                return_trg+=[sent_t.strip()]
    return return_src, return_trg


def load_or_extract_near_vocab(config_path,
                               model_path,
                               save_to,
                               save_to_full,
                               init_perturb_rate=0,
                               batch_size=50,
                               top_reserve=12,
                               all_with_UNK=False,
                               reload=True,
                               emit_as_id=False):
    """based on the embedding parameter from Encoder, extract near vocabulary for all words
    return: dictionary of vocabulary of near vocabs; and a the saved file
    :param config_path: (string) victim configs (for training data and vocabulary)
    :param model_path: (string) victim model path for trained embeddings
    :param save_to: (string) directory to store distilled near-vocab
    :param save_to_full: (string) directory to store full near-vocab
    :param init_perturb_rate: (float) the weight-adjustment for perturb
    :param batch_size: (integer) extract near vocab by batched cosine/Euclidean-similarity
    :param top_reserve: (integer) at most reserve top-k near candidates
    :param all_with_UNK: during generation, add UNK to all tokens as a candidate
    :param reload: reload from the save_to_path if previous record exists
    :param emit_as_id: (boolean) the key in return will be token ids instead of token
    """
    # load configs
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    data_configs = configs["data_configs"]
    model_configs = configs["model_configs"]
    # load vocabulary file
    src_vocab = Vocabulary(**data_configs["vocabularies"][0])

    # load embedding from model
    emb = nn.Embedding(num_embeddings=src_vocab.max_n_words,
                       embedding_dim=model_configs["d_word_vec"],
                       padding_idx=PAD
                       )
    model_params = torch.load(model_path, map_location="cpu")
    emb.load_state_dict({"weight": model_params["encoder.embeddings.embeddings.weight"]},
                        strict=True)
    len_mat = torch.sum(emb.weight**2, dim=1)**0.5  # length of the embeddings

    if os.path.exists(save_to) and reload:
        print("load from %s:" % save_to)
        return load_perturb_weight(save_to, src_vocab, emit_as_id)
    else:
        print("collect near candidates for vocabulary")
        avg_dist = 0
        avg_std = []
        counter = 0
        word2p = OrderedDict()
        word2near_vocab = OrderedDict()
        # omit similar vocabulary file (batched)
        with open(save_to, "w") as similar_vocab, open(save_to_full, "w") as full_similar_vocab:
            # every batched vocabulary collect average E-dist
            for i in range((src_vocab.max_n_words//batch_size)+1):
                if i*batch_size==src_vocab.max_n_words:
                    break

                index = torch.tensor(range(i*batch_size,
                              min(src_vocab.max_n_words, (i+1)*batch_size),
                              1))
                # extract embedding data
                slice_emb = emb(index)
                collect_len = torch.mm(len_mat.narrow(0, i * batch_size, min(src_vocab.max_n_words, (i+1)*batch_size)-i*batch_size).unsqueeze(1),
                                len_mat.unsqueeze(0))
                # filter top 10 nearest vocab, then filter with Eul-distance within certain range
                similarity = torch.mm(slice_emb,
                                       emb.weight.t()).div(collect_len)
                # get value and index
                topk_index = similarity.topk(top_reserve, dim=1)[1]

                sliceemb = slice_emb.unsqueeze(dim=1).repeat(1, top_reserve, 1)  # [batch_size, 1*8, dim]
                E_dist = ((emb(topk_index)-sliceemb)**2).sum(dim=-1)**0.5
                # print("avg Euclidean distance:", E_dist)
                avg_dist += E_dist.mean()
                avg_std += [E_dist.std(dim=1).mean()]
                counter += 1
            avg_dist = avg_dist.item() / counter
            # print(avg_dist)  # tensor object
            # print(avg_std)

            # output near candidates to file and return dictionary
            for i in range((src_vocab.max_n_words//batch_size)+1):
                if i*batch_size == src_vocab.max_n_words:
                    break
                index = torch.tensor(range(i*batch_size,
                              min(src_vocab.max_n_words, (i+1)*batch_size),
                              1))
                # extract embedding data
                slice_emb = emb(index)
                collect_len = torch.mm(len_mat.narrow(0, i * batch_size, min(src_vocab.max_n_words, (i+1)*batch_size)-i*batch_size).unsqueeze(1),
                                       len_mat.unsqueeze(0))
                # filter top k nearest vocab with cosine-similarity
                similarity = torch.mm(slice_emb,
                                      emb.weight.t()).div(collect_len)
                topk_val, topk_indices = similarity.topk(top_reserve, dim=1)
                # calculate E-dist
                sliceemb = slice_emb.unsqueeze(dim=1).repeat(1, top_reserve, 1)  # [batch_size, 1*topk, dim]
                E_dist = ((emb(topk_indices)-sliceemb)**2).sum(dim=-1)**0.5

                topk_val = E_dist.cpu().detach().numpy()
                topk_indices = topk_indices.cpu().detach().numpy()
                for j in range(topk_val.shape[0]):
                    bingo = 0.
                    src_word_id = j + i*batch_size

                    src_word = src_vocab.id2token(src_word_id)
                    near_vocab = []

                    similar_vocab.write(src_word + "\t")
                    full_similar_vocab.write(src_word + "\t")

                    # there is no candidates for reserved tokens
                    if src_word_id in [PAD, EOS, BOS, UNK]:
                        near_cand_id = src_word_id
                        near_cand = src_vocab.id2token(near_cand_id)

                        full_similar_vocab.write(near_cand + "\t")
                        similar_vocab.write(near_cand + "\t")
                        bingo = 1
                        if emit_as_id:
                            near_vocab += [near_cand_id]
                        else:
                            near_vocab += [near_cand]
                    else:
                        # extract near candidates according to cos-dist within averaged E-dist
                        for k in range(1, topk_val.shape[1]):
                            near_cand_id = topk_indices[j][k]
                            near_cand = src_vocab.id2token(near_cand_id)
                            full_similar_vocab.write(near_cand + "\t")
                            if topk_val[j][k] < avg_dist and (near_cand_id not in [PAD, EOS, BOS]):
                                bingo += 1
                                similar_vocab.write(near_cand + "\t")
                                if emit_as_id:
                                    near_vocab += [near_cand_id]
                                else:
                                    near_vocab += [near_cand]
                        # additionally add UNK as candidates
                        if bingo == 0 or all_with_UNK:
                            last_cand_ids = [UNK]
                            for final_reserve_id in last_cand_ids:
                                last_cand = src_vocab.id2token(final_reserve_id)
                                similar_vocab.write(last_cand + "\t")
                                if emit_as_id:
                                    near_vocab += [final_reserve_id]
                                else:
                                    near_vocab += [last_cand]

                    probability = bingo/(len(src_word)*top_reserve)
                    if init_perturb_rate != 0:
                        probability *= init_perturb_rate
                    similar_vocab.write("\t"+str(probability)+"\n")
                    full_similar_vocab.write("\t"+str(probability)+"\n")
                    if emit_as_id:
                        word2near_vocab[src_word_id] = near_vocab
                        word2p[src_word_id] = probability
                    else:
                        word2near_vocab[src_word] = near_vocab
                        word2p[src_word] = probability
        return word2p, word2near_vocab


# load the probability of randomization from existing files
def load_perturb_weight(save_to, src_vocab=None, emit_as_id=False):
    """
    random probability for the words.
    :param save_to: (string) saved files indicating top k most vocab
    :param src_vocab: Vocabulary class object (emit dictionary using token id)
    :param emit_as_id: (boolean) whether the dictionary use token ids
    :return: two dicts contains the random probability of dict <token: probability>
              <token: <candidate: probability>>
    """
    if emit_as_id:
        assert src_vocab is not None, "src_vocab must be provided when emit_as_id"
    with open(save_to) as similar_vocab:
        word2p = OrderedDict()
        word2near_vocab = OrderedDict()
        for line in similar_vocab:
            line = line.split("\t")
            if emit_as_id:
                word2near_vocab[src_vocab.token2id(line[0])] = [src_vocab.token2id(i) for i in line[1:-2]]
                word2p[src_vocab.token2id(line[0])] = float(line[-1])
            else:
                word2near_vocab[line[0]] = line[1:-2]
                word2p[line[0]] = float(line[-1])
    return word2p, word2near_vocab

# load translation model parameters
def load_translate_model(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)
    if "model" in state_dict:
        return state_dict["model"]
    return state_dict

def initial_random_perturb(config_path,
                           inputs,
                           w2p, w2vocab,
                           mode="len_based",
                           key_type="token",
                           show_bleu=False):
    """
    batched random perturb, perturb is based on random probability from the collected candidates
    meant to test initial attack rate.
    :param config_path: victim configs
    :param inputs: raw batched input (list) sequences in [batch_size, seq_len]
    :param w2p: indicates how likely a word is perturbed
    :param w2vocab: near candidates
    :param mode: based on word2near_vocab, how to distribute likelihood among candidates
    :param key_type: inputs are given by raw sequences of tokens or tokenized labels
    :param show_bleu: whether to show bleu of perturbed seqs (compare to original seqs)
    :return: list of perturbed inputs and list of perturbed flags
    """
    np.random.seed(int(time.time()))
    assert mode in ["uniform", "len_based"], "Mode must be in uniform or multinomial."
    assert key_type in ["token", "label"], "inputs key type must be token or label."
    # load configs
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    data_configs = configs["data_configs"]

    # load vocabulary file and tokenize
    src_vocab = Vocabulary(**data_configs["vocabularies"][0])
    perturbed_results = []
    flags = []
    for sent in inputs:
        if np.random.uniform() < 0.5:  # perturb
            perturbed_sent = []
            if key_type == "token":
                tokenized_sent = src_vocab.tokenizer.tokenize(sent)
                for word in tokenized_sent:
                    if np.random.uniform() < w2p[word]:
                        # need to perturb on lexical level
                        if mode == "uniform":
                            # uniform choose from candidates:
                            perturbed_sent += [w2vocab[word][np.random.choice(len(w2vocab[word]),
                                                                              1)[0]]]
                        elif mode == "len_based":
                            # weighted choose from candidates:
                            weights = [1./(1+abs(len(word)-len(c))) for c in w2vocab[word]]
                            norm_weights = [c/sum(weights) for c in weights]
                            perturbed_sent += [w2vocab[word][np.random.choice(len(w2vocab[word]),
                                                                              1,
                                                                              p=norm_weights
                                                                              )[0]]]
                    else:
                        perturbed_sent += [word]
                # print(perturbed_sent)  # yield same form of sequences of tokens
                perturbed_sent = src_vocab.tokenizer.detokenize(perturbed_sent)
            elif key_type == "label":  # tokenized labels
                for word_index in sent:
                    word = src_vocab.id2token(word_index)
                    if np.random.uniform() < w2p[word]:
                        if mode == "uniform":
                            # uniform choose from candidates:
                            perturbed_label = src_vocab.token2id(w2vocab[word][np.random.choice(
                                len(w2vocab[word]), 1
                            )[0]])
                            perturbed_sent += [perturbed_label]
                        elif mode == "len_based":
                            # weighted choose from candidates:
                            weights = [1. / (1 + abs(len(word) - len(c))) for c in w2vocab[word]]
                            norm_weights = [c / sum(weights) for c in weights]
                            perturbed_label = src_vocab.token2id(w2vocab[word][np.random.choice(len(w2vocab[word]),
                                                                             1,
                                                                             p=norm_weights
                                                                            )[0]])
                            perturbed_sent += [perturbed_label]
                    else:
                        perturbed_sent += [word_index]
            perturbed_results += [perturbed_sent]
            flags += [1]
            # out.write(perturbed_sent + "\n")
        else:
            perturbed_results += [sent]
            flags += [0]
    return perturbed_results, flags

def collect_pinyin(pinyin_path, src_path):
    # generate pinyin for every Chinese characters in training data
    """
    read from pinyin_path to generate pinyin dictionary
    :param pinyin_path: path to pin data file
    :param src_path: chinese src data path to collect
    :return: two dictionary of pinyin2char:{pinyin: [list of characters]},
                and char2pinyin: {ord(char): [list of pinyin]}
    """
    char2pyDict = {}
    py2charDict = {}
    count_char = {}
    for line in open(pinyin_path):
        k, v = line.strip().split('\t')
        char2pyDict[k] = v.split(" ")  # there can be multiple values(pinyin) for a key

    with open(src_path, "r") as input_src:
        line_counter = 0
        for line in input_src:
            line_counter += 1
            # if line_counter%1000 == 0:
            #     break
            # collect characters and their pinyin
            for char in line.strip():
                key = "%X" % ord(char)
                if char in count_char:
                    count_char[char] += 1
                else:
                    count_char[char] = 1
                try:
                    for pinyin in char2pyDict[key]:
                        pinyin = pinyin.strip()  # .lower()
                        if pinyin in py2charDict:
                            if char not in py2charDict[pinyin]:
                                py2charDict[pinyin].append(char)
                        else:
                            py2charDict[pinyin] = [char]
                except:  # special char without pinyin
                    continue
    return char2pyDict, py2charDict


def gen_UNK(src_token, vocab, char2pyDict, py2charDict):
    """
    when src_token is to be replaced by UNK, generate a token by randomly replace
    a character that has the same vocal (pinyin) as the src character
    (and make sure new token is UNK to vocab)
    if no UNK is found, return original token by default
    :param src_token: chinese src_token to be replaced by UNK
    :param vocab: data.vocabulary object to varify if result is UNK
    :param char2pyDict: dictionary {ord(char): pinyin}
    :param py2charDict: dictionary {pinyin}
    :return: a UNK word similar to src_token
    """
    edit_range = len(src_token)
    if src_token.endswith("@@"):  # don't break the signal for BPE
        edit_range -= 2

    if (char2pyDict is not None) and (py2charDict is not None):
        # generate homophone
        index = np.random.randint(edit_range)
        for i in range(edit_range):
            ori_char = src_token[index]
            new_token = src_token
            py_key = "%X" % ord(ori_char)
            if py_key in char2pyDict:
                # this character is available in gen_UNK
                for pinyin in char2pyDict[py_key]:
                    # check for every possible vocal
                    for candidate in py2charDict[pinyin]:
                        # check for every character share this vocal
                        new_token = list(new_token)
                        new_token[index] = candidate
                        new_token = "".join(new_token)
                        if candidate != ori_char and vocab.token2id(new_token) == UNK:
                            return new_token
                        else:
                            continue
            index = (index+1) % edit_range
    else:  # roman character replacement to generate unk
        # scramble the symble in between
        if edit_range > 3:
            index = np.random.randint(0, edit_range-2)
            new_token = src_token[:index] + \
                        src_token[index+1]+src_token[index]+\
                        src_token[index+2:]
            if vocab.token2id(new_token) == UNK:
                return new_token

    # nothing returned or token is too short, repeat last char
    char = src_token[edit_range - 1]
    token_stem = src_token[:edit_range]
    new_token = token_stem + char
    if src_token.endswith("@@"):
        temp_token = new_token+"@@"
    else:
        temp_token = new_token
    while vocab.token2id(temp_token) != UNK:
        new_token = new_token + char
        # print(src_token, "#>$#", new_token)
        if src_token.endswith("@@"):
            temp_token = new_token + "@@"
        else:
            temp_token = new_token
    new_token = temp_token
    # print(src_token, "-->", new_token)
    return new_token

def corpus_bleu_char(hyp_in, ref_in, need_tokenized=True):
    """
    preprocess corpus into char level and test BLEU,
    proposed to check modification rate
    :param hyp_in: files to be tested
    :param ref_in: reference file
    :param need_tokenized: for languages needs tokenization
    :return:
    """
    with open(hyp_in, "r") as hyp, open(ref_in, "r") as ref, \
            open("hyp_char", "w") as hyp_char, open("ref_char", "w") as ref_char:
        for line_hyp_in, line_ref_in in zip(hyp, ref):
            if not need_tokenized:
                line_hyp_in = line_hyp_in.replace(" ", "")
                line_ref_in = line_ref_in.replace(" ", "")
            hyp_char.write(" ".join(list(line_hyp_in)))
            ref_char.write(" ".join(list(line_ref_in)))
    # cat hyp_char | sacrebleu -lc --score-only  ref_char
    # sacrebleu_cmd = ["sacrebleu", "-l"] + ["--score-only",]+["ref_char"]
    cat = subprocess.Popen(("cat", "hyp_char"), stdout=subprocess.PIPE)
    cmd_bleu = subprocess.Popen(("/home/zouw/anaconda3/bin/sacrebleu", "-lc", "--score-only", "--force", "ref_char"),
                                stdin=cat.stdout,
                                stdout=subprocess.PIPE)
    bleu = cmd_bleu.communicate()[0].decode("utf-8").strip()
    print(bleu)
    bleu = float(bleu)
    subprocess.Popen("rm ref_char hyp_char", shell=True)
    return bleu


