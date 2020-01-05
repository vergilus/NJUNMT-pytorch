from adversarials.discriminator import TransDiscriminator
from adversarials.attacker import Attacker
from adversarials.adversarial_utils import *
from src.data.dataset import ZipDataset, TextLineDataset
from src.data.data_iterator import DataIterator
from src.utils.common_utils import Saver, Collections, should_trigger_by_steps
from src.utils.logging import *
from src.optim import *
from src.optim.lr_scheduler import *
from tensorboardX import SummaryWriter
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD

configs_path = [
    "/home/public_data/nmtdata/nmt-baselines/transformer-wmt15-enfr/small_baseline/transformer_wmt15_en2de.yaml",
    "/home/zouw/NJUNMT-pytorch/configs/transformer_nist_zh2en_bpe.yaml",
    "/home/zouw/pycharm_project_NMT_torch/configs/wmt_en2de_attack.yaml",
    "/home/zouw/pycharm_project_NMT_torch/configs/nist_zh2en_attack.yaml"]


def load_embedding(model, model_path, device):
    pretrained_params = torch.load(model_path, map_location=device)
    model.src_embedding.embeddings.load_state_dict(
        {"weight": pretrained_params["encoder.embeddings.embeddings.weight"]},
        strict=True)
    if "trg_embedding" in model.state_dict().keys():
        model.trg_embedding.embeddings.load_state_dict(
            {"weight": pretrained_params["decoder.embeddings.embeddings.weight"]},
            strict=True)
        INFO("load embedding from NMT model")
    return


def prepare_D_data(w2p, w2vocab, victim_config, seqs_x, seqs_y, use_gpu=False, batch_first=True):
    """
    Returns: return batched x,y tuples for training.
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
    # print(seqs_x)
    seqs_x, flags = initial_random_perturb(victim_config, seqs_x, w2p, w2vocab, key_type="label")
    # print(seqs_x)
    seqs_x = list(map(lambda s: [BOS] + s + [EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=PAD,
                         cuda=use_gpu, batch_first=batch_first)
    seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=PAD,
                         cuda=use_gpu, batch_first=batch_first)
    flags = torch.tensor(flags)
    if use_gpu:
        flags = flags.cuda()
    return x, y, flags


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


def compute_D_forward(discriminator_model, criterion,
                      seqs_x, seqs_y, gold_flags,
                      eval=False):
    """
    get discriminator loss
    :return: loss value
    """
    if not eval:
        # set components to training mode
        discriminator_model.train()
        criterion.train()
        with torch.enable_grad():
            class_probs = discriminator_model(seqs_x, seqs_y)
            loss = criterion(class_probs, gold_flags)
        torch.autograd.backward(loss)
        return loss.item()
    else:
        # set components to evaluation mode
        discriminator_model.eval()
        criterion.eval()
        with torch.no_grad():
            class_probs = discriminator_model(seqs_x, seqs_y)
            loss = criterion(class_probs, gold_flags)
    return loss.item()


def acc_validation(uidx, discriminator_model, valid_iterator,
                   victim_configs, w2p, w2vocab, batch_size, use_gpu):
    discriminator_model.eval()
    valid_iter = valid_iterator.build_generator(batch_size=batch_size)
    acc = 0
    sample_count = 0
    for batch in valid_iter:
        seq_nums, seqs_x, seqs_y = batch
        x, y, flags = prepare_D_data(w2p, w2vocab, victim_configs,
                                     seqs_x, seqs_y, use_gpu)
        # set components to evaluation mode
        discriminator_model.eval()
        with torch.no_grad():
            preds = discriminator_model(x, y).argmax(dim=-1)
            acc += torch.eq(preds, flags).sum()
            sample_count += preds.size(0)
    acc = acc.float()/sample_count
    return acc.item()


def test_discriminator(config_path,
                       save_to,
                       model_name="Discriminator",
                       shuffle=True,
                       use_gpu=True):
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    attack_configs = configs["attack_configs"]
    discriminator_model_configs = configs["discriminator_model_configs"]
    discriminator_optim_configs = configs["discriminator_optimizer_configs"]
    training_configs = configs["training_configs"]

    victim_config_path = attack_configs["victim_configs"]
    victim_model_path = attack_configs["victim_model"]
    with open(victim_config_path.strip()) as v_f:
        print("open victim configs...%s" % victim_config_path)
        victim_configs = yaml.load(v_f)
    data_configs = victim_configs["data_configs"]

    # building inputs
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_trg = Vocabulary(**data_configs["vocabularies"][1])
    # parallel data binding
    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0]),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_trg,
                        max_len=data_configs['max_len'][1]),
        shuffle=shuffle
    )
    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs["valid_data"][0],
                        vocabulary=vocab_src,
                        max_len=data_configs["max_len"][0]),
        TextLineDataset(data_path=data_configs["valid_data"][1],
                        vocabulary=vocab_trg,
                        max_len=data_configs["max_len"][1]),
        shuffle=shuffle
    )

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]
    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])
    # valid_iterator is bucketed by length to accelerate decoding (numbering to mark orders)
    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs["valid_batch_size"],
                                  use_bucket=True, buffer_size=50000, numbering=True)
    # initiate saver
    model_collections = Collections()
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(save_to, model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    # building model
    model_D = TransDiscriminator(n_src_words=vocab_src.max_n_words,
                                 n_trg_words=vocab_trg.max_n_words, **discriminator_model_configs)
    if use_gpu:
        model_D = model_D.cuda()
        CURRENT_DEVICE = "cuda"
    else:
        CURRENT_DEVICE = "cpu"
    # load embedding from trained NMT models
    load_embedding(model_D, model_path=victim_model_path, device=CURRENT_DEVICE)
    # TODO reloading parameters

    # classification need label smoothing to trigger Negative log likelihood loss
    criterion = nn.CrossEntropyLoss()
    # building optimizer
    optim = Optimizer(name=discriminator_optim_configs["optimizer"],
                      model=model_D,
                      lr=discriminator_optim_configs["learning_rate"],
                      grad_clip=discriminator_optim_configs["grad_clip"],
                      optim_args=discriminator_optim_configs["optimizer_params"])
    # Build scheduler for optimizer if needed
    if discriminator_optim_configs['schedule_method'] is not None:
        if discriminator_optim_configs['schedule_method'] == "loss":
            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **discriminator_optim_configs["scheduler_configs"]
                                                 )

        elif discriminator_optim_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **discriminator_optim_configs['scheduler_configs'])
        elif discriminator_optim_configs["schedule_method"] == "rsqrt":
            scheduler = RsqrtScheduler(optimizer=optim, **discriminator_optim_configs["scheduler_configs"])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(discriminator_optim_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    # reload latest checkpoint
    checkpoint_saver.load_latest(model=model_D, optim=optim, lr_scheduler=scheduler,
                                 collections=model_collections)

    # prepare training
    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    summary_writer = SummaryWriter(log_dir=save_to + "log")
    w2p, w2vocab = load_or_extract_near_vocab(config_path=victim_config_path,
                                              model_path=victim_model_path,
                                              init_perturb_rate=attack_configs["init_perturb_rate"],
                                              save_to=os.path.join(save_to, "near_vocab"),
                                              save_to_full=os.path.join(save_to, "full_near_vocab"),
                                              top_reserve=12)
    while True:  # infinite loop for training epoch
        training_iter = training_iterator.build_generator()
        for batch in training_iter:
            uidx += 1
            if discriminator_optim_configs["schedule_method"] is not None and discriminator_optim_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)
            # training session
            seqs_x, seqs_y = batch  # returned tensor type of the data
            optim.zero_grad()
            try:
                x, y, flags = prepare_D_data(w2p, w2vocab,
                                             victim_config_path,
                                             seqs_x, seqs_y, use_gpu=use_gpu)
                loss = compute_D_forward(model_D, criterion=criterion,
                                         seqs_x=x,
                                         seqs_y=y, gold_flags=flags)
                optim.step()
                print("loss:", loss)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory, skipping batch")
                    oom_count += 1
                    optim.zero_grad()
                else:
                    raise e
            # check for validation and save the model
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs["disp_freq"]):
                lrate = list(optim.get_lrate())[0]
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs["save_freq"]):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                checkpoint_saver.save(global_step=uidx, model=model_D, optim=optim,
                                      lr_scheduler=scheduler, collections=model_collections)
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs["loss_valid_freq"]):
                # validate average loss over samples on validation set
                n_sents = 0.
                sum_loss = 0.0
                valid_iter = valid_iterator.build_generator()
                for batch in valid_iter:
                    _, seqs_x, seqs_y = batch
                    n_sents += len(seqs_x)
                    x, y, flags = prepare_D_data(w2p, w2vocab,
                                                 victim_config_path,
                                                 seqs_x, seqs_y, use_gpu=use_gpu)
                    loss = compute_D_forward(model_D, criterion, x, y, gold_flags=flags, eval=True)
                    if np.isnan(loss):
                        WARN("NaN detected!")
                    sum_loss += float(loss)
                eval_loss = float(sum_loss / n_sents)
                summary_writer.add_scalar("valid", scalar_value=eval_loss, global_step=uidx)
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs["acc_valid_freq"]):
                # validate accuracy of the discriminator
                acc=acc_validation(uidx=uidx,
                                   discriminator_model=model_D, valid_iterator=valid_iterator,
                                   victim_configs=victim_config_path,
                                   w2p=w2p, w2vocab=w2vocab,
                                   batch_size=training_configs["acc_valid_batch_size"],
                                   use_gpu=use_gpu)
                summary_writer.add_scalar("accuracy", scalar_value=acc, global_step=uidx)

        eidx += 1
    pass


def test_attack(config_path,
                save_to,
                model_name="attacker",
                shuffle=True,
                use_gpu=True):
    """
    attack
    :param config_path: attack attack configs
    :param save_to: (string) saving directories
    :param model_name: (string) for saving names
    :param shuffle: (boolean) for batch scheme, shuffle data set
    :param use_gpu: (boolean) on gpu or not
    :return: attacked sequences
    """
    # initiate
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    attack_configs = configs["attack_configs"]
    attacker_model_configs = configs["attacker_model_configs"]
    attacker_optim_configs = configs["attacker_optimizer_configs"]
    training_configs = configs["training_configs"]

    victim_config_path = attack_configs["victim_configs"]
    victim_model_path = attack_configs["victim_model"]
    with open(victim_config_path.strip()) as v_f:
        print("open victim configs...%s" % victim_config_path)
        victim_configs = yaml.load(v_f)
    data_configs = victim_configs["data_configs"]

    # building inputs
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_trg = Vocabulary(**data_configs["vocabularies"][1])
    # parallel data binding
    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0]),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_trg,
                        max_len=data_configs['max_len'][1]),
        shuffle=shuffle
    )
    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs["valid_data"][0],
                        vocabulary=vocab_src,
                        max_len=data_configs["max_len"][0]),
        TextLineDataset(data_path=data_configs["valid_data"][1],
                        vocabulary=vocab_trg,
                        max_len=data_configs["max_len"][1]),
        shuffle=shuffle
    )

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]
    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])
    # valid_iterator is bucketed by length to accelerate decoding (numbering to mark orders)
    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs["valid_batch_size"],
                                  use_bucket=True, buffer_size=50000, numbering=True)

    # initiate saver
    model_collections = Collections()
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(save_to, model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    w2p, w2vocab = load_or_extract_near_vocab(config_path=victim_config_path,
                                              model_path=victim_model_path,
                                              init_perturb_rate=attack_configs["init_perturb_rate"],
                                              save_to=os.path.join(save_to, "near_vocab"),
                                              save_to_full=os.path.join(save_to, "full_near_vocab"),
                                              top_reserve=12,
                                              emit_as_id=True)
    # build attacker
    # attacker = Attacker(n_words=vocab_src.max_n_words,
    #                     **attacker_model_configs)
    # if use_gpu:
    #     attacker = attacker.cuda()
    #     CURRENT_DEVICE = "cuda"
    # else:
    #     CURRENT_DEVICE = "cpu"
    # load embedding from trained NMT models
    # load_embedding(attacker, model_path=victim_model_path, device=CURRENT_DEVICE)
    # attacker.eval()
    # for i in range(6):
    train_iter = training_iterator.build_generator()
    batch = train_iter.__next__()
    print(batch[1][3])
        # seqs_x, seqs_y = batch
        # x, y = prepare_data(seqs_x, seqs_y, use_gpu)
        # perturbed_x, flags = attacker.seq_attack(seqs_x_ids=x,
        #                                          w2p=w2p, w2vocab=w2vocab,
        #                                          training_mode=False)
        # print("origin:", x)
        # print("perturbed:", perturbed_x)



# test_discriminator(config_path=configs_path[2],
#                    save_to="./Discriminator_enfr",
#                    model_name="Discriminator")
def charF(ref_path, answ_path, beta=1):
    with open(ref_path, "r") as refs, open(answ_path, "r") as answ:
        # break down everything to char and calculate F_score
        collect_F1 = []
        intersect_count = 0.
        ref_count = 0.
        answ_count = 0.

        for ref_line, answ_line in zip(refs, answ):
            ref_line = [ word for word in ref_line.strip().split()]  # if word not in ["<UNK>", "<PAD>"]
            answ_line =[ word for word in answ_line.strip().split()]  #  if word not in ["<UNK>", "<PAD>"]
            ref_line = "".join(ref_line)
            answ_line = "".join(answ_line)
            ref_set = set(list(ref_line.strip()))
            answ_set = set(list(answ_line.strip()))
            print(ref_line, answ_line)
            intersect_count += len(ref_set.intersection(answ_set))
            ref_count+=len(ref_set)
            answ_count+=len(answ_set)

        total_p = intersect_count/answ_count
        total_r = intersect_count/ref_count
        # charF_score = 1/(0.5/total_p+0.5/total_r)
        charF_score = (1+beta**2)*total_p*total_r/(beta**2*total_p+total_r)
        return charF_score

def charBLEU(ref_path, answ_path):
    """
    corpus char level BLEU.
    :param ref_path: reference sequences file
    :param answ_path: answer sequence file
    :return:
    """
    hypothses = []
    referenfces = []
    with open(ref_path, "r") as refs, open(answ_path, "r") as answ:
        # print(len(refs), len(answ))
        # assert len(refs) == len(answ), "file mismatch!"
        for line1, line2 in zip(refs, answ):
            hypothses.append(list(line1.strip()))
            referenfces.append([list(line2.strip())])
    return corpus_bleu(referenfces, hypothses)

def mutualchrBLEU(ref_path, answ_path):
    hyp1 = []
    ref1 = []
    hyp0 = []
    ref0 = []
    with open(ref_path, "r") as refs, open(answ_path, "r") as answ:
        for line0, line1 in zip(refs, answ):
            line0 = list(line0.strip())
            line1 = list(line1.strip())
            hyp0.append(line0)
            ref0.append([line1])
            hyp1.append(line1)
            ref1.append([line0])
    mBLEU=(corpus_bleu(ref0, hyp0)+corpus_bleu(ref1, hyp1))/2
    return mBLEU

def WER(ref_path, answ_path, char_level=False):
    """
    calculate word error rates with levinshtein distance,
    get substitution / deletion / insertion w.r.t. total tokens in the reference
    :param ref_path: reference path
    :param answ_path: hypothesis path
    :param char_level: if test on char level (default False)
    :return:
    """

    edit_counter = 0
    ref_counter = 0
    with open(ref_path, "r") as refs, open(answ_path, "r") as answ:
        for ref_line, hyp_line in zip(refs, answ):
            # simple cleaning and tokenize by space
            ref_line = ref_line.strip().replace("\r", "").replace("\n", "")
            hyp_line = hyp_line.strip().replace("\r", "").replace("\n", "")
            if char_level:
                ref_line = list(ref_line)
                hyp_line = list(hyp_line)
            else:
                ref_line = ref_line.split()
                hyp_line = hyp_line.split()
            m = np.zeros((len(ref_line) + 1, len(hyp_line) + 1)).astype(dtype=np.int32)
            m[0, 1:] = np.arange(1, len(hyp_line) + 1)
            m[1:, 0] = np.arange(1, len(ref_line) + 1)
            # Now loop over remaining cell (from the second row and column onwards)
            # The value of each selected cell is:
            #
            #   if token represented by row == token represented by column:
            #       value of the top-left diagonal cell
            #   else:
            #       calculate 3 values:
            #            * top-left diagonal cell + 1 (which represents substitution)
            #            * left cell + 1 (representing deleting)
            #            * top cell + 1 (representing insertion)
            #       value of the smallest of the three
            #
            for i in range(1, m.shape[0]):
                for j in range(1, m.shape[1]):
                    if hyp_line[j - 1] == ref_line[i - 1]:
                        m[i, j] = m[i - 1, j - 1]
                    else:
                        m[i, j] = min(
                            m[i - 1, j - 1],
                            m[i, j - 1],
                            m[i - 1, j]
                        ) + 1
            edit_counter += m[len(ref_line), len(hyp_line)]
            ref_counter += len(ref_line)
    # print(edit_counter, ref_counter)
    wer = edit_counter/float(ref_counter)
    # and the minimum-edit distance is simply the value of the down-right most cell
    return wer

# zh2en
ref_path = "/home/public_data/nmtdata/nist_zh-en_1.34m/test/mt02.src"
# answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/random4_attack_zh2en_results/rand_mt05"
# answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/search_attack_tf_zh2en_word_results/search3_attack_mt06"
# answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/attack_en2fr_dl4mt_log/newstest2014/perturbed_src"
answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/attack_zh2en_dl4mt_log/perturbed_src"

# en2de
# ref_path = "/home/public_data/nmtdata/wmt14_en-de_data_selection/newstest2016.tok.en"
# # answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/search4_attack_tf_en2de_results/search4attack_newstest2016"
# answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/attack_en2de_dl4mt_log/newstest2016/perturbed_src"

# en2fr
# ref_path = "/home/public_data/nmtdata/wmt15_en-fr/test/newstest2014.en.tok"
# # answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/search4_attack_dl4mt_en2fr_results/search4attack_newstest2013"
# answ_path = "/home/zouw/pycharm_project_NMT_torch/adversarials/attack_en2fr_tf_log/perturbed_src"

print("chrF1=", charF(ref_path, answ_path, 1))
print("chrBLEU=", charBLEU(ref_path, answ_path))
print("WER=", WER(ref_path, answ_path))
print("chrWER=", WER(ref_path, answ_path, True))
