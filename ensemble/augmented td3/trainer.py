import sys
import utils
import torch
import numpy as np

class Trainer:
    def __init__(self, env, env_eval, agent, env_name, device, d_sample):

        self.device   = device
        self.agent    = agent
        self.env_name = env_name
        self.env      = env
        self.env_eval = env_eval
        self.d_sample = d_sample    # number of constant delayed sample

        self.start_step   = 10000
        self.update_after = 1000
        self.max_step     = 30000000
        self.batch_size   = 128
        self.update_every = 50
        self.policy_update_delay = 2

        self.eval_flag          = True
        self.eval_episode       = 5
        self.eval_freq          = 5000

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.finish_flag = False


    def evaluate(self):
        # Evaluate process
        reward_list = []

        for epi in range(self.eval_episode):
            epi_reward = 0
            state      = self.env_eval.reset()[0]
            state      = torch.FloatTensor(state)
            done       = False

            act_buf = []
            d = self.d_sample
            for _ in range(self.d_sample):
                act_buf.append(torch.FloatTensor([0]))

            while not done:
                I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor(act_buf[-d:]).unsqueeze(0)], dim=1)

                action = self.agent.get_action(I, add_noise=False)
                next_state, reward, terminated, truncated, _ = self.env_eval.step(act_buf[-d])
                next_state  = torch.FloatTensor(next_state)

                act_buf.append(torch.FloatTensor(action))

                done        = terminated or truncated
                epi_reward += reward
                state       = next_state

            reward_list.append(epi_reward)

        print("total_step {}  |  episode {}  |  Avg Reward {:.2f}  |  Max reward: {:.2f}  |  "
              "Min reward: {:.2f}".format(self.total_step, self.episode, sum(reward_list)/len(reward_list),
                                               max(reward_list), min(reward_list)))

    def run(self):
        while not self.finish_flag:
            self.episode       += 1
            self.episode_reward = 0
            self.local_step     = 0

            state = self.env.reset()[0]
            state = torch.FloatTensor(state)

            done = False

            act_buf = []
            d = self.d_sample
            for _ in range(self.d_sample):
                act_buf.append(torch.FloatTensor([0]))

            while not done:

                self.local_step += 1
                self.total_step += 1



                I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor(act_buf[-d:]).unsqueeze(0)], dim = 1)
                if self.total_step >= self.start_step:
                    action = np.array(self.agent.get_action(I, add_noise = True))
                    next_state, reward, terminated, truncated, _ = self.env.step(act_buf[-d])
                    next_state = torch.FloatTensor(next_state)
                else:
                    action = self.env.action_space.sample()
                    next_state, reward, terminated, truncated, _ = self.env.step(act_buf[-d])
                    next_state = torch.FloatTensor(next_state)

                act_buf.append(torch.FloatTensor(action))
                I_next  = torch.concatenate([next_state.unsqueeze(0), torch.FloatTensor(act_buf[-d:]).unsqueeze(0)], dim = 1)

                done = terminated or truncated
                self.episode_reward += reward

                done_mask = 0.0 if self.local_step == self.env._max_episode_steps else float(done)
                #self.agent.memory.push([state, action, reward, next_state, done_mask]) # buffer 1
                self.agent.memory.push(I, action, reward, I_next, done_mask) # buffer 2

                state = next_state

                # update
                if self.agent.memory.size >= self.batch_size and self.total_step >= self.update_after and self.total_step % self.update_every == 0:
                #if self.agent.memory.size() >= self.batch_size and self.total_step >= self.update_after and self.total_step % self.update_every == 0:
                    for i in range(self.update_every):
                        if (i+1) % self.policy_update_delay:
                            self.agent.train(option='both')  # train actor and critic both.
                        else:
                            self.agent.train(option='critic_only')  # train critic only

                # Evaluation.
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish_flag.
                if self.total_step == self.max_step:
                    self.finish_flag = True

