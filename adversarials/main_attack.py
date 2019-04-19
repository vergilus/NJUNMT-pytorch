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
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1,
                    help="parallel attacker process (default as 1)")
parser.add_argument("--config_path", type=str,
                    default="/home/zouw/pycharm_project_NMT_torch/configs/nist_zh2en_attack.yaml",
                    help="the path to attack config file.")
# parser.add_argument("--data_path", type=str, default=None,
#                     help="text_data for environments, default as None (for training Mode)")
parser.add_argument("--action_roll_steps", type=int, default=15,
                    help="training rolling steps (default as 15)")
parser.add_argument("--max_episode_lengths", type=int, default=200,
                    help="maximum steps for attack (default as 200)")
parser.add_argument("--max_episodes", type=int, default=500000,
                    help="maximum environment episode for training (default as 500k)")
parser.add_argument("--save_to", type=str, default="./attack_log",
                    help="the path for model-saving and log saving.")
parser.add_argument("--use_gpu", action="store_true", default=False,
                    help="Whether to use GPU.(default as false)")
parser.add_argument("--share_optim", action="store_true", default=False,
                    help="Whether to share optim across attackers (default as false)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default as 1)")

def run():
    # default threads as 1
    os.environ["OMP_NUM_THREADS"] = "1"

    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
    with open(args.config_path) as f:
        configs = yaml.load(f)
    attack_configs = configs["attack_configs"]
    attacker_model_configs = configs["attacker_configs"]["attacker_model_configs"]
    attacker_optimizer_configs = configs["attacker_configs"]["attacker_optimizer_configs"]
    # discriminator_data_configs = configs["discriminator_data_configs"]
    discriminator_configs = configs["discriminator_configs"]
    training_configs = configs["training_configs"]

    # initial checkpoint saver for global model
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(args.save_to, "A3Cmodel")),
                             num_max_keeping=training_configs["num_kept_checkpoints"])

    GlobalNames.SEED = training_configs["seed"]
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
        shuffle=True
    )  # this is basically a validation setting for translation
    data_iterator = DataIterator(dataset=data_set,
                                 batch_size=training_configs["valid_batch_size"],
                                 use_bucket=True,
                                 buffer_size=100000,
                                 numbering=True).build_generator()
    # global model variables (trg network)
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

    checkpoint_saver.load_latest(model=global_attacker,
                                 optim=optimizer,
                                 lr_scheduler=scheduler)

    if args.use_gpu:
        # collect available devices and distribute env on the available gpu
        device = "cuda"
        devices = []
        for i in range(torch.cuda.device_count()):
            devices += ["cuda:%d" % i]
    else:
        device = "cpu"
        devices = [device]

    process = []
    counter = mp.Value("i", 0)
    lock = mp.Lock()  # for multiple attackers update

    test(args.n, "cuda:0", args,
         attack_configs, discriminator_configs,
         src_vocab, trg_vocab, data_iterator,
         global_attacker, attacker_model_configs, counter)
    # train(0, device, args, counter, lock,
    #       attack_configs, discriminator_configs,
    #       src_vocab, trg_vocab, data_iterator,
    #       global_attacker, attacker_model_configs,
    #       attacker_optimizer_configs, optimizer, scheduler, checkpoint_saver)

    # # run the attack test for initiation
    # p = mp.Process(target=test,
    #                args=(args.n, "cuda:0", args,
    #                      attack_configs, discriminator_configs,
    #                      src_vocab, trg_vocab, data_iterator,
    #                      global_attacker, attack_configs,
    #                      counter)
    #                )
    # p.start()
    # process.append(p)
    # # run multiple training process of local attacker to update global one
    # for rank in range(args.n):
    #     p = mp.Process(target=train,
    #                    args=(rank, "cuda:%d" % (rank), args, counter, lock,
    #                          attack_configs, discriminator_configs,
    #                          src_vocab, trg_vocab, data_iterator,
    #                          global_attacker, attacker_model_configs,
    #                          attacker_optimizer_configs, optimizer, scheduler
    #                          checkpoint_saver ))
    #     p.start()
    #     process.append(p)
    # for p in process:
    #     p.join()


def test(rank, device, args,
         attack_configs, discriminator_configs,
         src_vocab, trg_vocab, data_iterator,
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
    :param data_iterator: shared by all envs and provides data set
    :param global_attacker: global attacker models
    :param attacker_configs: initiate local attacker model configs
    :param counter: multiprocessing counter
    :return:
    """
    torch.manual_seed(GlobalNames.SEED + rank)
    env = Translate_Env(attack_configs=attack_configs,
                        discriminator_configs=discriminator_configs,
                        src_vocab=src_vocab,
                        trg_vocab=trg_vocab,
                        data_iterator=data_iterator,
                        save_to=args.save_to, device="cuda")
    print("finish build env")

    # need a directory for saving and loading
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_to, "test_env"))

    local_attacker = attacker.Attacker(src_vocab.max_n_words,
                                       **attacker_configs)
    if device != "cpu":
        local_attacker.cuda()
    local_attacker.eval()

    episode_count = 0
    while True:
        padded_src = env.reset()
        #  sync with current attacker model
        local_attacker.load_state_dict(global_attacker.state_dict())
        episode_count += 1

        perturbed_x_ids = env.padded_src.clone()
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
                # value = local_attacker.get_critic(x=perturbed_x_ids, label=inputs)*actions
                target_of_step = []
                for batch_index in range(batch_size):
                    word_id = inputs[batch_index][1]
                    target_word_id = env.w2vocab[word_id.item()][np.random.choice(len(env.w2vocab[word_id.item()]), 1)[0]]
                    target_of_step += [target_word_id]
                # override the perturbed results with random choice from candidates
                perturbed_x_ids[:, t] *= (1 - actions)
                adjustification_ = torch.tensor(target_of_step, device=inputs.device)
                if GlobalNames.USE_GPU:
                    adjustification_ = adjustification_.cuda()
                perturbed_x_ids[:, t] += adjustification_ * actions
            # apply mask on the results
            perturbed_x_ids *= (1 - mask)
        # translate sequences and calculate degredated bleu scores on batches
        perturbed_result = env.translate(perturbed_x_ids)
        print("orgin_results:", env.seqs_y)
        print("perturbed_results:", perturbed_result)
        print("origin_bleu:", env.origin_bleu)

        # calculate final BLEU degredation:
        perturbed_bleu = []
        for i, sent in enumerate(env.seqs_y):
            # sentence is still surviving
            perturbed_bleu.append(
                bleu.sentence_bleu(references=[sent], hypothesis=perturbed_result[i]))
        print("perturbed_bleu: ", perturbed_bleu)
        bleu_degrade = (sum(env.origin_bleu)-sum(perturbed_bleu))/len(perturbed_bleu)
        summary_writer.add_scalar("bleu_degradation",
                                  scalar_value=bleu_degrade,
                                  global_step=episode_count)

        # edit BLEU
        edit_bleu = []
        padded_src = padded_src.tolist()
        perturbed_x_ids = perturbed_x_ids.cpu().numpy().tolist()
        print(padded_src)
        print(perturbed_x_ids)
        for i in range(len(padded_src)):
            src = [label for label in padded_src[i] if label != PAD]
            perturbed_src = [label for label in perturbed_x_ids[i] if label != PAD]
            edit_bleu += [bleu.sentence_bleu(references=[src], hypothesis=perturbed_src)]
        print("edit_bleu: ", edit_bleu)
        summary_writer.add_scalar("edit_bleu",
                                  scalar_value=sum(edit_bleu)/len(edit_bleu),
                                  global_step=episode_count)
        time.sleep(10)


def train(rank, device, args, counter, lock,
          attack_configs, discriminator_configs,
          src_vocab, trg_vocab, data_iterator,
          global_attacker, attacker_configs,
          attacker_optimizer_configs,
          optimizer=None, scheduler=None, saver=None, patience=2):
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
    :param attack_configs: attacker configurations
    :param discriminator_configs: discriminator settings
    :param src_vocab:
    :param trg_vocab:
    :param data_iterator: (data_iterator object) provide batched data labels
    :param global_attacker: the model to sync from
    :param attacker_configs: local attacker settings
    :param attacker_optimizer_configs: configs for attacker
    :param optimizer: uses shared optimizer for the attacker
            use local one if none
    :param scheduler: uses shared scheduler for the attacker,
            use local one if none
    :param model saver
    :return:
    """
    trust_acc = acc_bound = 0.85
    torch.manual_seed(GlobalNames.SEED + rank)
    env = Translate_Env(attack_configs=attack_configs,
                        discriminator_configs=discriminator_configs,
                        src_vocab=src_vocab, trg_vocab=trg_vocab,
                        data_iterator=data_iterator,
                        save_to=args.save_to, device=device)
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_to, "train_env%d" % rank))
    local_attacker = attacker.Attacker(src_vocab.max_n_words,
                                       **attacker_configs)

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

    padded_src = env.reset()
    padded_src = torch.from_numpy(padded_src)
    if device != "cpu":
        padded_src = padded_src.to(device)

    done = True
    episode_count = 0
    episode_length = 0
    local_steps = 0  # optimization steps
    discriminator_base_steps = local_steps
    patience_t = patience
    while True:
        # check for update of discriminator
        # if env.acc_validation(local_attacker, use_gpu=True if env.device != "cpu" else False) < 0.55:
        if episode_count % 150 == 0:
            while True:
                """ stop criterion:
                when updates a discriminator, we check for acc. If acc fails acc_bound,
                we reset the discriminator and try, until acc reaches the bound with patience.
                otherwise the training thread stops
                """
                discriminator_base_steps, trust_acc = env.update_discriminator(data_iterator,
                                                     local_attacker,
                                                     discriminator_base_steps,
                                                     min_update_steps=10,
                                                     max_update_steps=80,
                                                     accuracy_bound=acc_bound,
                                                     summary_writer=summary_writer)
                discriminator_base_steps += 1  # a flag to label the discriminator updates
                if trust_acc > 0.55:  # discriminator updated
                    break
                else:  # GAN target reached
                    INFO("Reset discriminator")
                    env.reset_discriminator()
                    patience_t -= 1
                    if patience_t == 0:
                        break
        if patience_t == 0:
            WARN("Training Thread should stop")
            break
        else:  # refresh patience
            patience_t = patience

        if saver and local_steps % 50 == 0:
            saver.save(global_step=local_steps,
                       model=global_attacker,
                       optim=optimizer,
                       lr_scheduler=scheduler)

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
        for i in range(args.action_roll_steps):
            episode_length += 1
            attack_out, critic_out = local_attacker(padded_src, padded_src[:, env.index-1:env.index+1])

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

            values.append(critic_out)  # list of torch variables (scalar)
            log_probs.append(log_attack_out)  # list of torch variables (scalar)
            rewards.append(reward)  # list of reward variables

            if done:
                episode_count += 1
                break

        R = torch.zeros(1, 1)
        gae = torch.zeros(1, 1)
        if device != "cpu":
            R = R.cuda()
            gae = gae.cuda()

        if not done:  # calculate value loss
            value = local_attacker.get_critic(padded_src, padded_src[:, env.index-1:env.index+1])
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0

        for i in reversed((range(len(rewards)))):
            R = attack_configs["gamma"] * R+rewards[i]
            advantage = R-values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + attack_configs["gamma"] * \
                      values[i+1]-values[i]
            gae = gae * attack_configs["gamma"] * attack_configs["tau"] +\
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
        (policy_loss + attack_configs["value_coef"] * value_loss).backward()

        if attacker_optimizer_configs["schedule_method"] is not None and attacker_optimizer_configs["schedule_method"] != "loss":
            scheduler.step(global_step=local_steps)

        # move the model params to CPU and
        # assign local gradients to the global model to update
        local_attacker.to("cpu").ensure_shared_grads(global_attacker)
        optimizer.step()
        print("bingo!")


if __name__ == "__main__":
    run()
