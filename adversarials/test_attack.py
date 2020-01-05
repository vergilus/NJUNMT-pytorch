from adversarials.adversarial_utils import *
from adversarials import attacker
from src.utils.logging import *
from src.utils.common_utils import *
from src.data.dataset import TextLineDataset
from src.data.data_iterator import DataIterator
from src.models import build_model
from src.decoding import beam_search

import argparse
import torch

parser = argparse.ArgumentParser()
#
parser.add_argument("--source_path", type=str, default="/home/public_data/nmtdata/nist_zh-en_1.34m/test/mt03.src", # /zouw/pycharm_project_NMT_torch/adversarials/attack_zh2en_tf_log/mt02/perturbed_src
                    help="the path for input files")
parser.add_argument("--model_path", type=str,
                    default="/home/zouw/pycharm_project_NMT_torch/adversarials/attack_zh2en_tf_log/ACmodel.final")
parser.add_argument("--config_path", type=str,
                    default="/home/zouw/pycharm_project_NMT_torch/configs/nist_zh2en_attack.yaml",
                    help="the path to attack config file.")
parser.add_argument("--save_to", type=str, default="/home/zouw/pycharm_project_NMT_torch/adversarials/attack_zh2en_tf_log",
                    help="the path for result saving.")
parser.add_argument("--batch_size", type=int, default=50,
                    help="test batch_size")
parser.add_argument("--unk_ignore", action="store_true", default=False,
                    help="Don't replace target words using UNK (default as false)")
parser.add_argument("--use_gpu", action="store_true", default=False,
                    help="Whether to use GPU.(default as false)")


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    pad seqs into torch tensor
    :param seqs_x:
    :param seqs_y:
    :param cuda:
    :param batch_first:
    :return:
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

def calculate_cummulate_survive(max_len, gamma, surrogate_step_survival):
    """
    estimate a overall surrogate survival values
    :param input: the src tensor to be attacked. shape: [batch, timestep]
    :param gamma: used in reinforced rewards
    :param surrogate_survival: surrogate single step survival rewards
    :return: a list of cummulated survival for every step,
    with estimate_accumulate_survive[timestep]=accumualted survive of sen_len "timestep"
    """
    estimate_accumulate_survive = [surrogate_step_survival]
    for i in range(1,max_len):
        estimate_accumulate_survive.append(
            estimate_accumulate_survive[i-1]*gamma+surrogate_step_survival
        )
    return torch.tensor(estimate_accumulate_survive)

def test_attack():
    """
    during test phrase, the attacker modifies inputs without constrains
    :return:
    """
    timer = Timer()
    args = parser.parse_args()
    with open(args.config_path) as f:
        configs = yaml.load(f)
    attack_configs = configs["attack_configs"]
    attacker_configs = configs["attacker_configs"]
    attacker_model_configs = attacker_configs["attacker_model_configs"]

    # for modification
    GlobalNames.SEED = attack_configs["seed"]
    torch.manual_seed(GlobalNames.SEED)
    # the Global variable of  USE_GPU is mainly used for environments
    GlobalNames.USE_GPU = args.use_gpu

    INFO("build vocabularies and data set")
    with open(attack_configs["victim_configs"], "r") as victim_f:
        victim_configs = yaml.load(victim_f)
    data_configs = victim_configs["data_configs"]
    src_vocab = Vocabulary(**data_configs["vocabularies"][0])
    trg_vocab = Vocabulary(**data_configs["vocabularies"][1])

    print("attack ", args.source_path)
    datset = TextLineDataset(data_path=args.source_path,
                             vocabulary=src_vocab)
    test_iterator = DataIterator(dataset=datset,
                                 batch_size=args.batch_size,
                                 use_bucket=attack_configs["use_bucket"],
                                 buffer_size=attack_configs["buffer_size"],
                                 numbering=True)
    total_amount = len(test_iterator)
    test_iterator = test_iterator.build_generator()
    _, w2vocab = load_or_extract_near_vocab(config_path=attack_configs["victim_configs"],
                                            model_path=attack_configs["victim_model"],
                                            init_perturb_rate=attack_configs["init_perturb_rate"],
                                            save_to=os.path.join(args.save_to, "near_vocab"),
                                            save_to_full=os.path.join(args.save_to, "full_near_vocab"),
                                            top_reserve=12,
                                            emit_as_id=True)
    if attack_configs["pinyin_data"] != "" and not args.unk_ignore:
        # for Chinese we adopt
        INFO("collect pinyin data for gen_UNK, this would take a while")
        char2pyDict, py2charDict = collect_pinyin(pinyin_path=attack_configs["pinyin_data"],
                                                  src_path=data_configs["train_data"][0])
    else:
        INFO("test without pinyin")
        char2pyDict, py2charDict = None, None

    INFO("build and reload attacker model parameters")
    global_attacker = attacker.Attacker(src_vocab.max_n_words,
                                        **attacker_model_configs)
    attacker_param = load_model_parameters(args.model_path)
    global_attacker.eval()
    global_attacker.load_state_dict(attacker_param)

    INFO("Build and reload translator...")
    nmt_model = build_model(n_src_vocab=src_vocab.max_n_words,
                            n_tgt_vocab=trg_vocab.max_n_words,
                            **victim_configs["model_configs"])
    nmt_model.eval()
    nmt_param = load_model_parameters(attack_configs["victim_model"])
    nmt_model.load_state_dict(nmt_param)
    if args.use_gpu:
        # collect available devices and distribute env on the available gpu
        global_attacker.cuda()
        nmt_model = nmt_model.cuda()

    result_indices = []  # to resume ordering
    origin_results = []  # original translation
    perturbed_seqs = []  # adversarial src
    perturbed_results = []  # adversarial translation
    overall_values = []  # attacker value estimation on first step: indicates overall degradation

    # translate all sentences and collect all adversarial src
    with open(os.path.join(args.save_to, "perturbed_src"), "w") as perturbed_src, \
         open(os.path.join(args.save_to, "perturbed_trans"), "w") as perturbed_trans, \
         open(os.path.join(args.save_to, "origin_trans"), "w") as origin_trans:
        i = 0
        timer.tic()
        for batch in test_iterator:
            i += 1
            if i:
                print(i * args.batch_size, "/", total_amount, " finished")
            numbers, seqs_x = batch
            # print(seqs_x)
            batch_size = len(seqs_x)
            x = prepare_data(seqs_x=seqs_x, cuda=args.use_gpu)
            x_mask = x.detach().eq(PAD).long()
            cummulate_survive = calculate_cummulate_survive(max_len=x.shape[1],
                                                            gamma=attack_configs["gamma"],
                                                            surrogate_step_survival=0)
            # x_len = (1 - x_mask).sum(dim=-1).float()

            with torch.no_grad():
                word_ids = beam_search(nmt_model=nmt_model, beam_size=5, max_steps=150,
                                       src_seqs=x, alpha=-1.0)
            word_ids = word_ids.cpu().numpy().tolist()  # in shape [batch_size, beam_size, max_len]
            # remove PAD and append result with its indices
            # we only take top-one final results from beam
            for sent_t in word_ids:
                top_result = [trg_vocab.id2token(wid) for wid in sent_t[0] if wid not in [PAD, EOS]]
                origin_results.append(trg_vocab.tokenizer.detokenize(top_result))
            result_indices += numbers

            # calculate adversarial value functions for each src position
            attack_results = []
            critic_results = []
            with torch.no_grad():
                for t in range(1, x.shape[1]-1):
                    attack_out, critic_out = global_attacker(x, label=x[:, t-1:t+1])
                    attack_results.append(attack_out.argmax(dim=1).unsqueeze(dim=1))
                    # print(mask_len.shape, critic_out.shape)
                    critic_results.append(critic_out)

            attack_results = torch.cat(attack_results, dim=1)
            temp_mask = (1-x_mask)[:, 1:x.shape[1]-1]
            attack_results *= temp_mask
            critic_results = torch.cat(critic_results, dim=1)*(1-x_mask)[:, 1:x.shape[1]-1].float()
            critic_results *= temp_mask.float()
            # critic_results = critic_results.cpu().numpy().tolist()
            # print(attack_results)
            # print(critic_results)

            # get adversarial samples for the src
            with torch.no_grad():
                perturbed_x_ids = x.clone().detach()
                batch_size, max_steps = x.shape
                for t in range(1, max_steps - 1):  # ignore BOS and EOS
                    inputs = x[:, t - 1:t + 1]
                    attack_out, critic_out = global_attacker(x=perturbed_x_ids, label=inputs)
                    actions = attack_out.argmax(dim=-1)
                    if t == 1:
                        overall_values += (critic_out - cummulate_survive[-t-2]).cpu().numpy().tolist()
                    # action is masked if the corresponding value estimation is negative
                    actions *= (critic_out-cummulate_survive[-t-2]).gt(0).squeeze().long()  #  - cummulate_survive[-t-2]
                    target_of_step = []
                    for batch_index in range(batch_size):
                        word_id = inputs[batch_index][1]
                        # select least similar candidate based on victim embedding
                        target_word_id = w2vocab[word_id.item()][0]  #[np.random.choice(len(w2vocab[word_id.item()]), 1)[0]]

                        # select nearest candidate based on victim embedding
                        # choose least similar candidates
                        # origin_emb = global_attacker.src_embedding(word_id)
                        # candidates_emb = global_attacker.src_embedding(torch.tensor(w2vocab[word_id.item()]).cuda())
                        # nearest = candidates_emb.matmul(origin_emb)\
                        #     .div((candidates_emb*candidates_emb).sum(dim=-1))\
                        #     .argmax(dim=-1).item()
                        # target_word_id = w2vocab[word_id.item()][nearest]

                        if args.unk_ignore and target_word_id == UNK:
                            # undo this attack if UNK is set to be ignored
                            target_word_id = word_id.item()
                        target_of_step += [target_word_id]
                    # override the perturbed results with choice from candidates
                    perturbed_x_ids[:, t] *= (1 - actions)
                    adjustification_ = torch.tensor(target_of_step, device=inputs.device)
                    if GlobalNames.USE_GPU:
                        adjustification_ = adjustification_.cuda()
                    perturbed_x_ids[:, t] += adjustification_ * actions
                # re-tokenization and validate UNK
                inputs = perturbed_x_ids.cpu().numpy().tolist()
                new_inputs = []
                for origin_indices, indices in zip(x.cpu().numpy().tolist(), inputs):
                    new_line_token = []  # for output files
                    # remove BOS, EOS, PAD, and detokenize to sentence
                    for origin_word_id, word_id in zip(origin_indices, indices):
                        if word_id not in [BOS, EOS, PAD]:
                            if word_id == UNK and origin_word_id != UNK:
                                # validate UNK induced by attack and append
                                new_line_token.append(gen_UNK(src_token=src_vocab.id2token(origin_word_id),
                                                              vocab=src_vocab,
                                                              char2pyDict=char2pyDict, py2charDict=py2charDict))
                            else:
                                new_line_token.append(src_vocab.id2token(word_id))
                    new_line_token = src_vocab.tokenizer.detokenize(new_line_token)
                    perturbed_seqs.append(new_line_token)
                    # tokenization must ignore original <UNK>
                    if not hasattr(src_vocab.tokenizer, "bpe"):
                        new_line = new_line_token.strip().split()
                    else:
                        new_token = []
                        for w in new_line_token.strip().split():
                            if w != src_vocab.id2token(UNK):
                                new_token.append(src_vocab.tokenizer.bpe.segment_word(w))
                            else:
                                new_token.append([w])
                        new_line = sum(new_token, [])
                    new_line = [src_vocab.token2id(t) for t in new_line]
                    new_inputs.append(new_line)
                # override perturbed_x_ids
                perturbed_x_ids = prepare_data(seqs_x=new_inputs,
                                               cuda=args.use_gpu)
                # batch translate perturbed_src
                word_ids = beam_search(nmt_model=nmt_model, beam_size=5, max_steps=150,
                                       src_seqs=perturbed_x_ids, alpha=-1.0)

            word_ids = word_ids.cpu().numpy().tolist()  # in shape [batch_size, beam_size, max_len]
            # translate adversarial inputs
            for sent_t in word_ids:
                top_result = [trg_vocab.id2token(wid) for wid in sent_t[0] if wid not in [PAD, EOS]]
                perturbed_results.append(trg_vocab.tokenizer.detokenize(top_result))

        print(timer.toc(return_seconds=True), "sec")
        # resume original ordering and output to files
        origin_order = np.argsort(result_indices).tolist()
        for line in [origin_results[ii] for ii in origin_order]:
            origin_trans.write(line+"\n")
        for line, value in [(perturbed_seqs[ii], overall_values[ii]) for ii in origin_order]:
            perturbed_src.write(line+"\n")  # +" "+str(value)
        for line in [perturbed_results[ii] for ii in origin_order]:
            perturbed_trans.write(line+"\n")


if __name__ == "__main__":
    test_attack()
