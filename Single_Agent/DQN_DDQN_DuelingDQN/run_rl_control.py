import argparse
import json
import logging
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import cityflow
from cityflow_env import CityFlowEnv
from utility import parse_roadnet, plot_data_lists
from dqn_agent import DQNAgent, DDQNAgent
from duelingDQN import DuelingDQNAgent

from keras.callbacks import TensorBoard # New

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes

def main():
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/global_config.json', help='config file')
    parser.add_argument('--algo', type=str, default='DQN', choices=['DQN', 'DDQN', 'DuelDQN'], help='choose an algorithm')
    parser.add_argument('--inference', action="store_true", help='inference or training')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--epoch', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--num_step', type=int, default=1000, help='number of timesteps for one episode, and for inference')
    parser.add_argument('--save_freq', type=int, default=50, help='model saving frequency')
    parser.add_argument('--batch_size', type=int, default=30, help='batchsize for training')
    parser.add_argument('--phase_step', type=int, default=3, help='time of one phase')

    args = parser.parse_args()

    # preparing config
    # # for environment
    config = json.load(open(args.config))
    config["num_step"] = args.num_step


    # config["replay_data_path"] = "replay"
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    # # for agent
    # intersection_id = list(config['lane_phase_info'].keys())[0]
    config["intersection_id"] = "intersection_1_1"
    config["state_size"] = len(config['lane_phase_info'][config["intersection_id"]]['start_lane']) + 1
    # config["state_size"] = len(config['lane_phase_info'][config["intersection_id"]]['start_lane']) + 1 + 4

    phase_list = config['lane_phase_info'][config["intersection_id"]]['phase']
    config["action_size"] = len(phase_list)
    config["batch_size"] = args.batch_size

    logging.info(phase_list)

    model_dir = "model/{}_{}".format(args.algo, date)
    result_dir = "result/{}_{}".format(args.algo, date)
    config["result_dir"] = result_dir

    # build agent
    if args.algo == 'DQN':
        agent = DQNAgent(config)
    elif args.algo == 'DDQN':
        agent = DDQNAgent(config)
    elif args.algo == 'DuelDQN':
        agent = DuelingDQNAgent(config)

    # parameters for training and inference
    # batch_size = 32
    EPISODES = args.epoch
    learning_start = 200 # Main/Online model starts learning (.fit()) after 200 episodes.
    update_model_freq = args.batch_size # Main model is fit() every 'update_model_freq' steps.
    update_target_model_freq = 200 # Target model is updated with weights of main model every 200 steps.

    if not args.inference:
        # build cityflow environment
        cityflow_config["saveReplay"] = False
        json.dump(cityflow_config, open(config["cityflow_config_file"], 'w'))
        env = CityFlowEnv(config)

        # make dirs
        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists("result"):
            os.makedirs("result")
        os.makedirs(model_dir)
        os.makedirs(result_dir)

        # training
        total_step = 0
        episode_rewards = []
        episode_scores = []
        with tqdm(total=EPISODES*args.num_step) as pbar:

            for i in range(EPISODES):

                # Update tensorboard step every episode
                agent.tensorboard.step = i+1

                # print("episode: {}".format(i))
                env.reset()
                # agent.epsilon=0.9
                state = env.get_state()

                episode_length = 0
                episode_reward = 0
                episode_score = 0
                while episode_length < args.num_step:

                    action = agent.choose_action(state) # agent chooses action randomly or based on max q value.
                    action_phase = phase_list[action] # phase to trigger.
                    # no yellow light
                    next_state, reward = env.step(action_phase) # one step
                    # last_action_phase = action_phase
                    episode_length += 1
                    total_step += 1
                    episode_reward += reward
                    episode_score += env.get_score()

                    for _ in range(args.phase_step-1): # default value of phase_step = 3
                        next_state, reward_ = env.step(action_phase)
                        reward += reward_

                    reward /= args.phase_step

                    pbar.update(1)
                    # store to replay buffer
                    agent.remember(state, action_phase, reward, next_state)

                    state = next_state

                    # training
                    # if total_step > learning_start and total_step % update_model_freq == 0:
                    #     agent.replay() # fit() the main model

                    if total_step > learning_start:
                        if episode_length == args.num_step:
                            terminal_state = True
                        else:
                            terminal_state = False
                        agent.replay(terminal_state) # fit() the main model

                    # update target Q netwark
                    if total_step > learning_start and total_step % update_target_model_freq == 0:
                        agent.update_target_network()

                    # logging
                    # logging.info("\repisode:{}/{}, total_step:{}, action:{}, reward:{}"
                    #             .format(i+1, EPISODES, total_step, action, reward))
                    pbar.set_description(
                        "total_step:{}, episode:{}, episode_step:{}, reward:{}".format(total_step, i+1, episode_length, reward))


                # save episode rewards
                episode_rewards.append(episode_reward)
                episode_scores.append(episode_score)

                if not (i+1) % AGGREGATE_STATS_EVERY or (i+1) == 1:
                    average_reward = sum(episode_rewards[-AGGREGATE_STATS_EVERY:])/len(episode_rewards[-AGGREGATE_STATS_EVERY:])
                    min_reward = min(episode_rewards[-AGGREGATE_STATS_EVERY:])
                    max_reward = max(episode_rewards[-AGGREGATE_STATS_EVERY:])
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=agent.epsilon)

                    # Save model, but only when min reward is greater or equal a set value
                    if min_reward >= MIN_REWARD:
                        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

                print("score: {}, mean reward:{}".format(episode_score, episode_reward/args.num_step))


                # save model
                if (i + 1) % args.save_freq == 0:
                    if args.algo != 'DuelDQN':
                        agent.model.save(model_dir + "/{}-{}.h5".format(args.algo, i+1))
                    else:
                        agent.save(model_dir + "/{}-ckpt".format(args.algo), i+1)

                    # save reward to file
                    df = pd.DataFrame({"rewards": episode_rewards})
                    df.to_csv(result_dir + '/rewards.csv', index=None)

                    df = pd.DataFrame({"rewards": episode_scores})
                    df.to_csv(result_dir + '/scores.csv', index=None)


            # save figure
            plot_data_lists([episode_rewards], ['episode reward'], figure_name=result_dir + '/rewards.pdf')
            plot_data_lists([episode_scores], ['episode score'], figure_name=result_dir + '/scores.pdf')


    else:
        # inference
        cityflow_config["saveReplay"] = True
        json.dump(cityflow_config, open(config["cityflow_config_file"], 'w'))
        env = CityFlowEnv(config)
        env.reset()

        agent.load(args.ckpt)

        state = env.get_state()
        scores = []
        for i in range(args.num_step):
            action = agent.choose_action(state) # index of action
            action_phase = phase_list[action] # actual action
            next_state, reward = env.step(action_phase) # one step

            for _ in range(args.phase_step - 1):
                next_state, reward_ = env.step(action_phase)
                reward += reward_

            reward /= args.phase_step

            scores.append(env.get_score())
            state = next_state

            # logging
            logging.info("step:{}/{}, action:{}, reward:{}".format(i+1, args.num_step, action, reward))

        df = pd.DataFrame({"scores": scores})
        df.to_csv(result_dir + '/scores.csv', index=None)
        plot_data_lists([scores], ['scores'], figure_name=result_dir + '/scores.pdf')



if __name__ == '__main__':
    main()
