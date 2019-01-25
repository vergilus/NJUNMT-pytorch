from adversarials.discriminator import TransDiscriminator
from adversarials.adversarial_utils import *
from src.data.dataset import ZipDataset, TextLineDataset
from src.data.data_iterator import DataIterator
from src.utils.common_utils import Saver, Collections, should_trigger_by_steps
from src.utils.logging import *
from src.optim import *
from src.optim.lr_scheduler import *
from src.modules.criterions import NMTCriterion
from tensorboardX import SummaryWriter
import numpy as np

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD

models_path = [
    "/home/public_data/nmtdata/nmt-baselines/transformer-wmt15-enfr-bpe32k/small_baseline/baseline_enfr_save/transformer.best.final",
    "/home/zouw/NJUNMT-pytorch/scripts/save_zhen_bpe/transformer.best.final"]
configs_path = [
    "/home/public_data/nmtdata/nmt-baselines/transformer-wmt15-enfr-bpe32k/small_baseline/transformer_wmt15_en2fr.yaml",
    "/home/zouw/NJUNMT-pytorch/configs/transformer_nist_zh2en_bpe.yaml",
    "/home/zouw/pycharm_project_NMT_torch/configs/nist_zh2en_attack.yaml"]


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
                       victim_model_path,
                       victim_config_path,
                       model_name="Discriminator",
                       shuffle=True,
                       use_gpu=True):
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    data_configs = configs["data_configs"]
    model_configs = configs["model_configs"]
    optim_configs = configs["optimizer_configs"]
    training_configs = configs["training_configs"]

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

    train_batch_size = training_configs["batch_size"] * max(1, training_configs["update_cycle"])
    train_buffer_size = training_configs["buffer_size"] * max(1, training_configs["update_cycle"])
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
    best_model_prefix = os.path.join(save_to, model_name + "best")
    model_collections = Collections()
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(save_to, model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    # building model
    model_D = TransDiscriminator(n_src_words=vocab_src.max_n_words,
                                 n_trg_words=vocab_trg.max_n_words, **model_configs)
    if use_gpu:
        model_D = model_D.cuda()
        CURRENT_DEVICE = "cuda:0"
    else:
        CURRENT_DEVICE = "cpu"
    # load embedding from trained NMT models
    load_embedding(model_D, model_path=victim_model_path, device=CURRENT_DEVICE)
    # TODO reloading parameters

    # classification need label smoothing to trigger Negative log likelihood loss
    criterion = nn.CrossEntropyLoss()
    # building optimizer
    optim = Optimizer(name=optim_configs["optimizer"],
                      model=model_D,
                      lr=optim_configs["learning_rate"],
                      grad_clip=optim_configs["grad_clip"],
                      optim_args=optim_configs["optimizer_params"])
    # Build scheduler for optimizer if needed
    if optim_configs['schedule_method'] is not None:
        if optim_configs['schedule_method'] == "loss":
            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **optim_configs["scheduler_configs"]
                                                 )

        elif optim_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optim_configs['scheduler_configs'])
        elif optim_configs["schedule_method"] == "rsqrt":
            scheduler = RsqrtScheduler(optimizer=optim, **optim_configs["scheduler_configs"])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optim_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    # reload latest checkpoint
    checkpoint_saver.load_latest(model=model_D, optim=optim, lr_scheduler=scheduler,
                                 collections=model_collections)

    # prepare training
    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    summary_writer = SummaryWriter(log_dir=save_to + model_name + "log")
    w2p, w2vocab = load_or_extract_near_vocab(config_path=victim_config_path,
                                              model_path=victim_model_path,
                                              save_to="similar_vocab",
                                              save_to_full="full_similar_vocab",
                                              top_reserve=12)
    while True:  # infinite loop for training epoch
        training_iter = training_iterator.build_generator()
        for batch in training_iter:
            uidx += 1
            if optim_configs["schedule_method"] is not None and optim_configs["schedule_method"] != "loss":
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


test_discriminator(config_path=configs_path[2],
                   save_to="./",
                   victim_model_path=models_path[1],
                   victim_config_path=configs_path[1],
                   model_name="Discriminator")
