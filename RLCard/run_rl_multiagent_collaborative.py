import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
            'game_num_players': 3
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        # agent1 = NFSPAgent(
        #     num_actions=env.num_actions,
        #     state_shape=env.state_shape[0],
        #     hidden_layers_sizes=[64,64],
        #     q_mlp_layers=[64,64],
        #     device='cuda:0',
        # )
        # agent2 = NFSPAgent(
        #     num_actions=env.num_actions,
        #     state_shape=env.state_shape[0],
        #     hidden_layers_sizes=[64,64],
        #     q_mlp_layers=[64,64],
        #     device='cuda:0',
        # )
        # agent3 = NFSPAgent(
        #     num_actions=env.num_actions,
        #     state_shape=env.state_shape[0],
        #     hidden_layers_sizes=[64,64],
        #     q_mlp_layers=[64,64],
        #     device='cuda:0',
        # )

    # agent1 = torch.load('/content/drive/MyDrive/CS6284/agent1.pth')
    # agent2 = torch.load('/content/drive/MyDrive/CS6284/agent2.pth')
    agent1 = torch.load(r'./checkpoint_collab/collab_nfsp_friend1.pth')
    agent2 = torch.load(r'./checkpoint_collab/collab_nfsp_friend2.pth')
    agent3 = torch.load(r'./checkpoint_collab/collab_nfsp_comp2.pth')
    
    agents = [agent1, agent2, agent3]
    # for _ in range(1, env.num_players):
    #     agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()
                agents[1].sample_episode_policy()
                # agents[2].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for i in range(2):
              for ts in trajectories[i]:
                agents[i].feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        args.num_eval_games,
                    )
                )
            if episode % 5000 == 0:
              save_path1 = os.path.join(args.log_dir, f'{episode}_collab_nfsp_friend1.pth')
              save_path2 = os.path.join(args.log_dir, f'{episode}_collab_nfsp_friend2.pth')
            #   save_path3 = os.path.join(args.log_dir, f'{episode}_collab_nfsp_comp3.pth')
              
              torch.save(agent1, save_path1)
              torch.save(agent2, save_path2)
            #   torch.save(agent3, save_path3)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    # plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path1 = os.path.join(args.log_dir, 'collab_nfsp_friend1.pth')
    save_path2 = os.path.join(args.log_dir, 'collab_nfsp_friend2.pth')
    # save_path3 = os.path.join(args.log_dir, 'collab_nfsp_comp3.pth')
    
    torch.save(agent1, save_path1)
    torch.save(agent2, save_path2)
    # torch.save(agent3, save_path3)
    print('Model saved in', save_path1)
    
    # return csv_path, fig_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='limit-holdem-collaborative',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='nfsp',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=25000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./checkpoint_collab',
    )

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)