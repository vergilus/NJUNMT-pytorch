"""
this is the baseline search attack with given ratio of random positions.
"""
import yaml
import torch
import copy
import numpy as np
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.models import build_model
from src.modules.criterions import Criterion
from adversarials.adversarial_utils import gen_UNK, collect_pinyin

PAD = Vocabulary.PAD
BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
UNK = Vocabulary.UNK
ratio = 0.4
victim_model_path = "/home/user_data/zouw/Models/save_tf_zh2en_bpe/transformer.best.final"
victim_config_path = "/home/user_data/zouw/Models/transformer_nist_zh2en_bpe.yaml"
test_file_path = ["/home/public_data/nmtdata/nist_zh-en_1.34m/test/mt05.src",
                  "/home/public_data/nmtdata/nist_zh-en_1.34m/test/mt05.ref0"]
test_file_path_temp = ["mt02.zh.tmp",
                       "mt02.en.tmp"]
near_candidates_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/attack_zh2en_tf_log/near_vocab"
pinyin_path = "/home/user_data/zouw/chnchar2pinyin.dat"
output_path = "search_attack_mt05_nounk"
use_gpu = True
unk_ignore = True

def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.
    Returns:
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


class AdvCriterion(Criterion):

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        ompute input gradient change
        :param inputs: source sequences logarithm probabilities in shape [... , V]
        :param labels: target labels index tensor in shape[... ,]
        :param kwargs:
        :return: sum of the probability on the target tokens
        """
        batch_size = labels.size(0)
        scores = inputs.view(-1, inputs.size(-1))
        num_token = scores.size(-1)

        gtruth = labels.view(-1)  # flatten ground truth labels
        tdata = gtruth.detach()
        mask = torch.nonzero(tdata.eq(PAD).squeeze())
        one_hot = torch.zeros(1, num_token)
        if labels.is_cuda:
            one_hot = one_hot.cuda()
        tmp_ = one_hot.repeat(gtruth.size(0), 1)  # inflate gtruth to the same size of inputs
        tmp_.scatter_(1, tdata.unsqueeze(1), 1)
        if mask.numel() > 0:
            tmp_.index_fill_(0, mask, 0)
        gtruth = tmp_.detach()
        trg_log_prob = (gtruth * (1 - scores))    # dim=-1
        return trg_log_prob.sum(dim=-1)

# load from candidates
print("load near candidates")
near_cand = dict()
with open(near_candidates_path,"r") as candidate_file:
    for line in candidate_file:
        data4line = line.strip().split("\t")
        src_token = data4line[0]
        cand_tokens = data4line[1:-2]
        near_cand[src_token] = cand_tokens

print("load configuration")
with open(victim_config_path.strip()) as f:
    victim_configs = yaml.load(f)
data_configs = victim_configs["data_configs"]
model_configs = victim_configs["model_configs"]
print("build vocabulary")
vocab_src = Vocabulary(**data_configs["vocabularies"][0])
vocab_trg = Vocabulary(**data_configs["vocabularies"][1])

if not unk_ignore and pinyin_path != "":
    # load pin if there is any
    # for Chinese we adopt
    print("collect pinyin data for gen_UNK, this would take a while")
    char2pyDict, py2charDict = collect_pinyin(pinyin_path=pinyin_path,
                                              src_path=data_configs["train_data"][0])
else:
    print("test without pinyin")
    char2pyDict, py2charDict = None, None

valid_dataset = ZipDataset(
    TextLineDataset(data_path=test_file_path[0],
                    vocabulary=vocab_src),
    TextLineDataset(data_path=test_file_path[1],
                    vocabulary=vocab_trg),
)
valid_iterator = DataIterator(dataset=valid_dataset,
                              batch_size=1,
                              use_bucket=True, buffer_size=100000, numbering=True)
valid_iterator = valid_iterator.build_generator()

nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                        n_tgt_vocab=vocab_trg.max_n_words, **model_configs)
nmt_model.train()
loss_adv = AdvCriterion()
if use_gpu:
    nmt_model = nmt_model.cuda()
    loss_adv = loss_adv.cuda()

# reload parameters
params = load_model_parameters(victim_model_path, map_location="cpu")
nmt_model.load_state_dict(params)


def get_input_gradient(x, y):
    """
    :param x: tensor variables
    :param y: tensor variables
    :return:
    """
    nmt_model.zero_grad()  # clear gradient remains
    loss_adv(inputs=nmt_model(x, y), labels=y).backward()
    for param_name, param_weight in nmt_model.named_parameters():
        if param_name == "encoder.embeddings.embeddings.weight":
            # print(param_weight.grad.shape)
            return param_weight.grad.sum()
    return None


random_perturbed_out = dict()  # {number: sequences}
for batch in valid_iterator:
    number, seqs_x, seqs_y = batch
    x, y = prepare_data(seqs_x, seqs_y, cuda=use_gpu)
    print(number, x.shape[-1], y.shape[-1], int(x.shape[-1]*ratio))
    # sample random positions uniformly based on given ratio
    length = len(seqs_x[0])
    positions_to_attack = np.random.choice(length, int(length * ratio), [1.0/length] * length).tolist()
    positions_to_attack.sort()

    # calculate criterion and get corresponding input gradient
    original_graident = get_input_gradient(x, y)

    overall_value = 0  # overall search loss

    perturbed_seqs_x = copy.deepcopy(seqs_x)
    if len(positions_to_attack) > 0:
        print("position to attack:", positions_to_attack)
    else:
        print("too short, skip")
        perturbed_src_tokens = []
        for origin_src_id in seqs_x[0]:
            perturbed_src_tokens.append(vocab_src.id2token(origin_src_id))
        random_perturbed_out[number[0]] = vocab_src.tokenizer.detokenize(perturbed_src_tokens)
        continue

    for position in positions_to_attack:
        src_token_id = seqs_x[0][position]
        src_token = vocab_src.id2token(src_token_id)
        if vocab_src.id2token(src_token_id) in near_cand:
            # temp_seqs_x = np.repeat(perturbed_x, len(near_cand[src_token]), axis=0)
            # temp_seqs_y = np.repeat(seqs_y, len(near_cand[src_token]), axis=0)
            # for i in range(len(near_cand[src_token])):
            #     temp_seqs_x[i][position] = vocab_src.token2id(near_cand[src_token][i])
            # temp_x, temp_y = prepare_data(temp_seqs_x, temp_seqs_y, cuda=use_gpu)
            # delta_grad = get_input_gradient(temp_x, temp_y)-original_graident
            # best_index = torch.argmax(delta_grad)
            # print(best_index.item())
            final_cand_id = 0
            max_delta_gradient = 0
            for cand in near_cand[vocab_src.id2token(src_token_id)]:
                temp_seqs_x = perturbed_seqs_x
                temp_seqs_x[0][position] = vocab_src.token2id(cand)
                perturbed_x, y = prepare_data(temp_seqs_x, seqs_y, cuda=use_gpu)
                delta_grad = torch.abs(get_input_gradient(perturbed_x, y)-original_graident)
                delta_grad *= torch.abs(perturbed_x-x).sum()
                if delta_grad.item() > max_delta_gradient:
                    max_delta_gradient = delta_grad.item()
                    final_cand_id = vocab_src.token2id(cand)

            if final_cand_id == 0:  # it's meaningless to attack
                perturbed_seqs_x[0][position] = src_token_id
            else:  # apply attack
                if unk_ignore and final_cand_id == UNK:  # check for unk
                    perturbed_seqs_x[0][position] = src_token_id
                else:
                    perturbed_seqs_x[0][position] = final_cand_id
        else:  # ignore origin UNK
            continue

    perturbed_src_tokens = []
    for origin_src_id, perturbed_src_id in zip(seqs_x[0], perturbed_seqs_x[0]):
        # print(vocab_src.id2token(origin_src_id), vocab_src.id2token(perturbed_src_id))
        if origin_src_id != UNK and perturbed_src_id == UNK:
            # print(vocab_src.id2token(origin_src_id))
            perturbed_src_tokens.append(gen_UNK(src_token=vocab_src.id2token(origin_src_id),
                                                vocab=vocab_src,
                                                char2pyDict=char2pyDict, py2charDict=py2charDict))
        else:
            # if perturbed_src_id not in [BOS, EOS, PAD]:
            perturbed_src_tokens.append(vocab_src.id2token(perturbed_src_id))
    print(vocab_src.tokenizer.detokenize(perturbed_src_tokens))
    random_perturbed_out[number[0]] = vocab_src.tokenizer.detokenize(perturbed_src_tokens)

with open(output_path, "w") as random_result:
    for i in range(len(random_perturbed_out)):
        random_result.write(random_perturbed_out[i]+"\n")
