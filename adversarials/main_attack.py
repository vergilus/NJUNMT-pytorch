from adversarials.adversarial_utils import *
from adversarials import attacker
from adversarials.trans_env import Translate_Env
from src.optim import Optimizer
from src.optim.lr_scheduler import RsqrtScheduler, ReduceOnPlateauScheduler, NoamScheduler
from src.utils.logging import *
from src.utils.common_utils import *
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.data_iterator import DataIterator
from tensorboardX import SummaryWriter

import nltk.translate.bleu_score as bleu
import argparse
import torch
import torch.multiprocessing as _mp



# "/home/zouw/pycharm_project_NMT_torch/configs/nist_zh2en_attack.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1,
                    help="parallel attacker process (default as 1)")
parser.add_argument("--config_path", type=str,
                    default="/home/zouw/pycharm_project_NMT_torch/configs/attack_zh2en.yaml",
                    help="the path to attack config file.")
parser.add_argument("--save_to", type=str, default="./attack_en2de_dl4mt_log",
                    help="the path for model-saving and log saving.")
parser.add_argument("--action_roll_steps", type=int, default=15,
                    help="training rolling steps (default as 15)")
parser.add_argument("--max_episode_lengths", type=int, default=200,
                    help="maximum steps for attack (default as 200)")
parser.add_argument("--max_episodes", type=int, default=500000,
                    help="maximum environment episode for training (default as 500k)")
parser.add_argument("--use_gpu", action="store_true", default=False,
                    help="Whether to use GPU.(default as false)")
parser.add_argument("--share_optim", action="store_true", default=False,
                    help="Whether to share optim across attackers (default as false)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default as 1)")

def run():
    # default actor threads as 1
    os.environ["OMP_NUM_THREADS"] = "1"
    mp = _mp.get_context('spawn')
    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
    with open(args.config_path, "r") as f,\
         open(os.path.join(args.save_to, "current_attack_configs.yaml"), "w") as current_configs:
        configs = yaml.load(f)
        yaml.dump(configs, current_configs)
    attack_configs = configs["attack_configs"]
    attacker_configs = configs["attacker_configs"]
    attacker_model_configs = attacker_configs["attacker_model_configs"]
    attacker_optimizer_configs = attacker_configs["attacker_optimizer_configs"]
    discriminator_configs = configs["discriminator_configs"]
    # training_configs = configs["training_configs"]

    # initial checkpoint saver and best saver for global model
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(args.save_to, "ACmodel")),
                             num_max_keeping=attack_configs["num_kept_checkpoints"])
    GlobalNames.SEED = attack_configs["seed"]
    # the Global variable of  USE_GPU is mainly used for environments
    GlobalNames.USE_GPU = args.use_gpu
    torch.manual_seed(GlobalNames.SEED)

    # build vocabulary and data iterator for env
    with open(attack_configs["victim_configs"], "r") as victim_f:
        victim_configs = yaml.load(victim_f)
    data_configs = victim_configs["data_configs"]
    src_vocab = Vocabulary(**data_configs["vocabularies"][0])
    trg_vocab = Vocabulary(**data_configs["vocabularies"][1])
    data_set = ZipDataset(
        TextLineDataset(data_path=data_configs["train_data"][0],
                        vocabulary=src_vocab,),
        TextLineDataset(data_path=data_configs["train_data"][1],
                        vocabulary=trg_vocab,),
        shuffle=attack_configs["shuffle"]
    )  # we build the parallel data sets and iterate inside a thread

    # global model variables (trg network to save the results)
    global_attacker = attacker.Attacker(src_vocab.max_n_words,
                                        **attacker_model_configs)
    global_attacker = global_attacker.cpu()
    global_attacker.share_memory()
    if args.share_optim:
        # initiate optimizer and set to share mode
        optimizer = Optimizer(name=attacker_optimizer_configs["optimizer"],
                              model=global_attacker,
                              lr=attacker_optimizer_configs["learning_rate"],
                              grad_clip=attacker_optimizer_configs["grad_clip"],
                              optim_args=attacker_optimizer_configs["optimizer_params"])
        optimizer.optim.share_memory()
        # Build scheduler for optimizer if needed
        if attacker_optimizer_configs['schedule_method'] is not None:
            if attacker_optimizer_configs['schedule_method'] == "loss":
                scheduler = ReduceOnPlateauScheduler(optimizer=optimizer,
                                                     **attacker_optimizer_configs["scheduler_configs"])
            elif attacker_optimizer_configs['schedule_method'] == "noam":
                scheduler = NoamScheduler(optimizer=optimizer, **attacker_optimizer_configs['scheduler_configs'])
            elif attacker_optimizer_configs["schedule_method"] == "rsqrt":
                scheduler = RsqrtScheduler(optimizer=optimizer, **attacker_optimizer_configs["scheduler_configs"])
            else:
                WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(
                    attacker_optimizer_configs['schedule_method']))
                scheduler = None
        else:
            scheduler = None
    else:
        optimizer = None
        scheduler = None

    # load from checkpoint: only agent
    checkpoint_saver.load_latest(model=global_attacker,
                                 optim=optimizer,
                                 lr_scheduler=scheduler)

    if args.use_gpu:
        # collect available devices and distribute env on the available gpu
        device = "cuda"
        devices = []
        for i in range(torch.cuda.device_count()):
            devices += ["cuda:%d" % i]
        print("available gpus:", devices)
    else:
        device = "cpu"
        devices = [device]

    process = []
    counter = mp.Value("i", 0)
    lock = mp.Lock()  # for multiple attackers update

    train(0, device, args, counter, lock,
          attack_configs, discriminator_configs,
          src_vocab, trg_vocab, data_set,
          global_attacker, attacker_configs,
          optimizer, scheduler,
          checkpoint_saver)

    valid(args.n, device, args,
         attack_configs, discriminator_configs,
         src_vocab, trg_vocab, data_set,
         global_attacker, attacker_configs, counter)

    # run multiple training process of local attacker to update global one
    # for rank in range(args.n):
    #     print("initialize training thread on cuda:%d" % (rank+1))
    #     p = mp.Process(target=train,
    #                    args=(rank, "cuda:%d" % (rank+1), args, counter, lock,
    #                          attack_configs, discriminator_configs,
    #                          src_vocab, trg_vocab, data_set,
    #                          global_attacker, attacker_configs,
    #                          optimizer, scheduler,
    #                          checkpoint_saver))
    #     p.start()
    #     process.append(p)
    # # run the dev thread for initiation
    # print("initialize dev thread on cuda:0")
    # p = mp.Process(target=valid,
    #                args=(0, "cuda:0", args,
    #                      attack_configs, discriminator_configs,
    #                      src_vocab, trg_vocab, data_set,
    #                      global_attacker, attacker_configs,
    #                      counter)
    #                )
    # p.start()
    # process.append(p)
    #
    # for p in process:
    #     p.join()


def valid(rank, device, args,
          attack_configs, discriminator_configs,
          src_vocab, trg_vocab, data_set,
          global_attacker, attacker_configs,
          counter):
    """
    for test thread (runs the network results) we simply runs the attacker
    on batch of sequences on the environment using current global attaker
     (without discriminator, since discriminator is not general)
    :param rank: rank of the thread (might be multi-processing)
    :param device: running device for the ac network thread
    :param args: global args
    :param attack_configs: initiate env
    :param discriminator_configs: initiate env
    :param src_vocab: initiate env
    :param trg_vocab: initiate env
    :param data_set: provides data set for iterator
    :param global_attacker: global attacker models
    :param attacker_configs: initiate local attacker configs
    :param counter: multiprocessing counter
    :return:
    """
    torch.manual_seed(GlobalNames.SEED + rank)
    attacker_model_configs = attacker_configs["attacker_model_configs"]
    valid_iterator = DataIterator(dataset=data_set,
                                  batch_size=attack_configs["batch_size"],
                                  use_bucket=attack_configs["use_bucket"],
                                  buffer_size=attack_configs["buffer_size"],
                                  numbering=True)
    valid_iterator = valid_iterator.build_generator()
    env = Translate_Env(attack_configs=attack_configs,
                        discriminator_configs=discriminator_configs,
                        src_vocab=src_vocab,
                        trg_vocab=trg_vocab,
                        data_iterator=valid_iterator,
                        save_to=args.save_to, device=device)
    INFO("finish building validation env")
    # need a directory for saving and loading
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_to, "dev_env"))

    local_attacker = attacker.Attacker(src_vocab.max_n_words,
                                       **attacker_model_configs)
    if device != "cpu":
        local_attacker = local_attacker.to(device)
    local_attacker.eval()

    def trans_from_vocab(vocab, ids):
        """
        transcribe from vocabulary
        :param vocab: A Vocabulary object
        :param ids: 2D list of ids, in shape [sent, tokens]
        :return: 2D list of tokens, with all special tokens removed (detokenized)
        """
        result = []
        for sent_ids in ids:
            result += [vocab.ids2sent([i for i in sent_ids if i not in [PAD, EOS, BOS]])]
        return result

    episode_count = 0
    with open(os.path.join(args.save_to, "dev_env/src_enhanced"), "w") as src_f, \
         open(os.path.join(args.save_to, "dev_env/src_pert"), "w") as src_pert, \
         open(os.path.join(args.save_to, "dev_env/trg_enhanced"), "w") as trg_f, \
         open(os.path.join(args.save_to, "dev_env/trans_origin"), "w") as trans_origin,\
         open(os.path.join(args.save_to, "dev_env/trans_pert"), "w") as trans_pert:
        while True:
            padded_src = env.reset()
            #  sync with current attacker model
            local_attacker.load_state_dict(global_attacker.state_dict())
            episode_count += 1

            perturbed_x_ids = env.padded_src.clone().detach()
            # print(perturbed_x_ids)
            mask = perturbed_x_ids.detach().eq(PAD).long()
            # print(mask)
            with torch.no_grad():
                batch_size, max_steps = padded_src.shape
                for t in range(1, max_steps - 1):  # ignore BOS and EOS
                    inputs = env.padded_src[:, t - 1:t + 1]
                    attack_out = local_attacker.get_attack(x=perturbed_x_ids, label=inputs)
                    actions = attack_out.argmax(dim=-1)
                    actions_entropy = -(attack_out * torch.log(attack_out)).sum(dim=-1).mean()
                    summary_writer.add_scalar("action_entropy", scalar_value=actions_entropy.item(), global_step=episode_count)
                    target_of_step = []
                    for batch_index in range(batch_size):
                        word_id = inputs[batch_index][1]
                        # random choice from candidates
                        target_word_id = env.w2vocab[word_id.item()][np.random.choice(len(env.w2vocab[word_id.item()]), 1)[0]]
                        target_of_step += [target_word_id]
                    # override the perturbed results with random choice from candidates
                    perturbed_x_ids[:, t] *= (1 - actions)
                    adjustification_ = torch.tensor(target_of_step, device=inputs.device)
                    if GlobalNames.USE_GPU:
                        adjustification_ = adjustification_.to(device)
                    perturbed_x_ids[:, t] += adjustification_ * actions
                # apply mask on the results
                perturbed_x_ids *= (1 - mask)
            # translate sequences and calculate degredated bleu scores on batches
            perturbed_result = env.translate(perturbed_x_ids)

            trg_y = trans_from_vocab(trg_vocab, env.seqs_y)
            trans_y_p = trans_from_vocab(trg_vocab, perturbed_result)
            trans_y = trans_from_vocab(trg_vocab, env.origin_result)
            print("golden:", trg_y)
            print("origin_results:", trans_y)
            print("perturbed_results:", trans_y_p)

            # calculate final BLEU degredation:
            perturbed_bleu = []
            for i, sent in enumerate(env.seqs_y):
                # sentence is still surviving
                perturbed_bleu.append(
                    bleu.sentence_bleu(references=[sent], hypothesis=perturbed_result[i], emulate_multibleu=True))
            print("origin_bleu:", env.origin_bleu)
            print("perturbed_bleu: ", perturbed_bleu)
            bleu_degrade = (sum(env.origin_bleu)-sum(perturbed_bleu))/len(perturbed_bleu)

            summary_writer.add_scalar("bleu_degradation",
                                      scalar_value=bleu_degrade,
                                      global_step=episode_count)

            # edit BLEU
            edit_bleu = []
            padded_src = padded_src.tolist()
            perturbed_x_ids = perturbed_x_ids.cpu().numpy().tolist()

            trans_x = trans_from_vocab(src_vocab, padded_src)
            trans_x_p = trans_from_vocab(src_vocab, perturbed_x_ids)
            print(trans_x)
            print(trans_x_p)

            for i in range(len(padded_src)):
                src = [label for label in padded_src[i] if label != PAD]
                perturbed_src = [label for label in perturbed_x_ids[i] if label != PAD]
                edit_bleu += [bleu.sentence_bleu(references=[src], hypothesis=perturbed_src, emulate_multibleu=True)]
            print("edit_bleu: ", edit_bleu)
            summary_writer.add_scalar("edit_bleu",
                                      scalar_value=sum(edit_bleu)/len(edit_bleu),
                                      global_step=episode_count)

            # output enhanced results to log files
            for i in range(len(perturbed_bleu)):
                if perturbed_bleu[i] > env.origin_bleu[i]:
                    src_f.write(trans_x[i] + "\n")
                    src_pert.write(trans_x_p[i] + "\n")
                    trg_f.write(trg_y[i] + "\n")
                    trans_origin.write(trans_y[i] + "\n")
                    trans_pert.write(trans_y_p[i] + "\n")

            time.sleep(5)


def train(rank, device, args, counter, lock,
          attack_configs, discriminator_configs,
          src_vocab, trg_vocab, data_set,
          global_attacker, attacker_configs,
          optimizer=None, scheduler=None, saver=None):
    """
    running train process
    #1# train the env_discriminator
    #2# run attacker AC based on rewards from trained env_discriminator
    #3# run training updates attacker AC
    #4#
    :param rank: (int) the rank of the process (from multiprocess)
    :param device: the device of the process
    :param counter: python multiprocess variable
    :param lock: python multiprocess variable
    :param args: global args
    :param attack_configs: attack settings
    :param discriminator_configs: discriminator settings
    :param src_vocab:
    :param trg_vocab:
    :param data_set: (data_iterator object) provide batched data labels
    :param global_attacker: the model to sync from
    :param attacker_configs: local attacker settings
    :param optimizer: uses shared optimizer for the attacker
            use local one if none
    :param scheduler: uses shared scheduler for the attacker,
            use local one if none
    :param saver: model saver
    :return:
    """
    trust_acc = acc_bound = discriminator_configs["acc_bound"]
    converged_bound = discriminator_configs["converged_bound"]
    patience = discriminator_configs["patience"]
    attacker_model_configs = attacker_configs["attacker_model_configs"]
    attacker_optimizer_configs = attacker_configs["attacker_optimizer_configs"]

    torch.manual_seed(GlobalNames.SEED + rank)

    attack_iterator = DataIterator(dataset=data_set,
                                   batch_size=attack_configs["batch_size"],
                                   use_bucket=True,
                                   buffer_size=attack_configs["buffer_size"],
                                   numbering=True)

    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_to, "train_env%d" % rank))
    local_attacker = attacker.Attacker(src_vocab.max_n_words,
                                       **attacker_model_configs)
    # build optimizer for attacker
    if optimizer is None:
        optimizer = Optimizer(name=attacker_optimizer_configs["optimizer"],
                              model=global_attacker,
                              lr=attacker_optimizer_configs["learning_rate"],
                              grad_clip=attacker_optimizer_configs["grad_clip"],
                              optim_args=attacker_optimizer_configs["optimizer_params"])
        # Build scheduler for optimizer if needed
        if attacker_optimizer_configs['schedule_method'] is not None:
            if attacker_optimizer_configs['schedule_method'] == "loss":
                scheduler = ReduceOnPlateauScheduler(optimizer=optimizer,
                                                     **attacker_optimizer_configs["scheduler_configs"]
                                                     )
            elif attacker_optimizer_configs['schedule_method'] == "noam":
                scheduler = NoamScheduler(optimizer=optimizer, **attacker_optimizer_configs['scheduler_configs'])
            elif attacker_optimizer_configs["schedule_method"] == "rsqrt":
                scheduler = RsqrtScheduler(optimizer=optimizer, **attacker_optimizer_configs["scheduler_configs"])
            else:
                WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(
                    attacker_optimizer_configs['schedule_method']))
                scheduler = None
        else:
            scheduler = None

    attacker_iterator = attack_iterator.build_generator()
    env = Translate_Env(attack_configs=attack_configs,
                        discriminator_configs=discriminator_configs,
                        src_vocab=src_vocab, trg_vocab=trg_vocab,
                        data_iterator=attacker_iterator,
                        save_to=args.save_to, device=device)
    episode_count = 0
    episode_length = 0
    local_steps = 0  # optimization steps: for learning rate schedules
    patience_t = patience
    while True:  # infinite loop of data set
        # we will continue with a new iterator with refreshed environments
        # whenever the last iterator breaks with "StopIteration"
        attacker_iterator = attack_iterator.build_generator()
        env.reset_data_iter(attacker_iterator)
        padded_src = env.reset()
        padded_src = torch.from_numpy(padded_src)
        if device != "cpu":
            padded_src = padded_src.to(device)
        done = True
        discriminator_base_steps = local_steps

        while True:
            # check for update of discriminator
            # if env.acc_validation(local_attacker, use_gpu=True if env.device != "cpu" else False) < 0.55:
            if episode_count % attacker_configs["attacker_update_steps"] == 0:
                """ stop criterion:
                when updates a discriminator, we check for acc. If acc fails acc_bound,
                we reset the discriminator and try, until acc reaches the bound with patience.
                otherwise the training thread stops
                """
                try:
                    discriminator_base_steps, trust_acc = env.update_discriminator(
                        local_attacker,
                        discriminator_base_steps,
                        min_update_steps=discriminator_configs[
                            "acc_valid_freq"],
                        max_update_steps=discriminator_configs[
                            "discriminator_update_steps"],
                        accuracy_bound=acc_bound,
                        summary_writer=summary_writer)
                except StopIteration:
                    INFO("finish one training epoch, reset data_iterator")
                    break

                discriminator_base_steps += 1  # a flag to label the discriminator updates
                if trust_acc < converged_bound:  # GAN target reached
                    patience_t -= 1
                    INFO("discriminator reached GAN convergence bound: %d times" % patience_t)
                else:  # reset patience if discriminator is refreshed
                    patience_t = patience

            if saver and local_steps % attack_configs["save_freq"] == 0:
                saver.save(global_step=local_steps,
                           model=global_attacker,
                           optim=optimizer,
                           lr_scheduler=scheduler)

                if trust_acc < converged_bound and patience_t == patience-1:
                    # we only save the first params reaching acc_bound, because the latter one
                    # tends to deteriorate in exploration.
                    torch.save(global_attacker.state_dict(), os.path.join(args.save_to, "ACmodel.final"))

            if patience_t == 0:
                WARN("maximum patience reached. Training Thread should stop")
                break

            local_attacker.train()  # switch back to training mode

            # for a initial (reset) attacker from global parameters
            if done:
                INFO("sync from global model")
                local_attacker.load_state_dict(global_attacker.state_dict())
            # move the local attacker params back to device after updates
            local_attacker = local_attacker.to(device)
            values = []  # training critic: network outputs
            log_probs = []
            rewards = []  # actual rewards
            entropies = []

            local_steps += 1
            # run sequences step of attack
            try:
                for i in range(args.action_roll_steps):
                    episode_length += 1
                    attack_out, critic_out = local_attacker(padded_src, padded_src[:, env.index - 1:env.index + 1])
                    logit_attack_out = torch.log(attack_out)
                    entropy = -(attack_out * logit_attack_out).sum(dim=-1).mean()

                    summary_writer.add_scalar("action_entropy", scalar_value=entropy, global_step=local_steps)
                    entropies.append(entropy)  # for entropy loss
                    actions = attack_out.multinomial(num_samples=1).detach()
                    # only extract the log prob for chosen action (avg over batch)
                    log_attack_out = logit_attack_out.gather(-1, actions).mean()
                    padded_src, reward, terminal_signal = env.step(actions.squeeze())
                    done = terminal_signal or episode_length > args.max_episode_lengths

                    with lock:
                        counter.value += 1

                    if done:
                        episode_length = 0
                        padded_src = env.reset()

                    padded_src = torch.from_numpy(padded_src)
                    if device != "cpu":
                        padded_src = padded_src.to(device)

                    values.append(critic_out.mean())  # list of torch variables (scalar)
                    log_probs.append(log_attack_out)  # list of torch variables (scalar)
                    rewards.append(reward)  # list of reward variables

                    if done:
                        episode_count += 1
                        break
            except StopIteration:
                INFO("finish one training epoch, reset data_iterator")
                break

            R = torch.zeros(1, 1)
            gae = torch.zeros(1, 1)
            if device != "cpu":
                R = R.to(device)
                gae = gae.to(device)

            if not done:  # calculate value loss
                value = local_attacker.get_critic(padded_src, padded_src[:, env.index - 1:env.index + 1])
                R = value.mean().detach()

            values.append(R)
            policy_loss = 0
            value_loss = 0

            # collect values for training
            for i in reversed((range(len(rewards)))):
                # value loss and policy loss must be clipped to stabilize training
                R = attack_configs["gamma"] * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                delta_t = rewards[i] + attack_configs["gamma"] * \
                          values[i + 1] - values[i]
                gae = gae * attack_configs["gamma"] * attack_configs["tau"] + \
                      delta_t
                policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                              attack_configs["entropy_coef"] * entropies[i]
                print("policy_loss", policy_loss)
                print("gae", gae)

            # update with optimizer
            optimizer.zero_grad()
            # we decay the loss according to discriminator's accuracy as a trust region constrain
            summary_writer.add_scalar("policy_loss", scalar_value=policy_loss * trust_acc, global_step=local_steps)
            summary_writer.add_scalar("value_loss", scalar_value=value_loss * trust_acc, global_step=local_steps)

            total_loss = trust_acc * policy_loss + \
                    trust_acc * attack_configs["value_coef"] * value_loss
            total_loss.backward()

            if attacker_optimizer_configs["schedule_method"] is not None and attacker_optimizer_configs[
                "schedule_method"] != "loss":
                scheduler.step(global_step=local_steps)

            # move the model params to CPU and
            # assign local gradients to the global model to update
            local_attacker.to("cpu").ensure_shared_grads(global_attacker)
            optimizer.step()
            print("bingo!")

        if patience_t == 0:
            INFO("Reach maximum Discriminator patience, Finish")
            break


if __name__ == "__main__":
    run()
