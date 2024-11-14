import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    try:
        # Safely remove the log directory if needed
        if os.path.isdir(opt.log_path):
            shutil.rmtree(opt.log_path)
    except PermissionError as e:
        print(f"Warning: {e}. Skipping log directory cleanup.")

    # Create the log directory (skip error if it already exists)
    os.makedirs(opt.log_path, exist_ok=True)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    
    # Initialize Q network and target network
    model = DeepQNetwork()
    target_model = DeepQNetwork()  # Target network
    target_model.load_state_dict(model.state_dict())  # Initialize target network with same weights as Q network
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        target_model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    t1 = time.time()
    total_time = 0
    best_score = 1000

    while epoch < opt.num_epochs:
        start_time = time.time()
        next_steps = env.get_next_states()

        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]  # Q network predictions for action selection
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()  # Use Q network to choose action

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=True)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue

        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        epoch += 1

        # Sample a batch from replay memory
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)  # Q values from Q network

        # DDQN specific: Use Q network for action selection, but target network for Q-value estimation
        model.eval()
        target_model.eval()
        with torch.no_grad():
            next_q_values = model(next_state_batch)  # Q network for action selection
            next_actions = torch.argmax(next_q_values, dim=1)  # Actions chosen by Q network
            target_q_values = target_model(next_state_batch)  # Target network for Q-value estimation
        model.train()
        target_model.train()

        # Compute y_batch: if done, use reward; else, use Q-target
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * target_q_values[i, action]
                  for i, (reward, done, action) in enumerate(zip(reward_batch, done_batch, next_actions))))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        # Update target network every 10 epochs
        if epoch % 5 == 0:
            target_model.load_state_dict(model.state_dict())

        end_time = time.time()
        use_time = end_time - t1 - total_time
        total_time = end_time - t1
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}, Used time: {}, total used time: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines,
            use_time,
            total_time))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            print("save interval model: {}".format(epoch))
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
        elif final_score>best_score:
            best_score = final_score
            print("save best model: {}".format(best_score))
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, best_score))


if __name__ == "__main__":
    opt = get_args()
    train(opt)

