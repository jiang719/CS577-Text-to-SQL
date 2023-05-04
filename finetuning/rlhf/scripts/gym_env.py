import gym
from gym import spaces
import torch

class TextToSQLEnv(gym.Env):
    def __init__(self, dataset, tokenizer, max_length=1536):
        super(TextToSQLEnv, self).__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Action space: a single token from the tokenizer's vocabulary
        self.action_space = spaces.Discrete(len(tokenizer))

        # Observation space: a sequence of tokens (indices) of length max_length
        self.observation_space = spaces.Box(low=0, high=len(tokenizer)-1, shape=(max_length,), dtype=int)

    def reset(self):
        self.current_sample = self.dataset[np.random.choice(len(self.dataset))]
        input_ids = self.current_sample['input_ids']
        self.current_state = input_ids
        return input_ids

    def step(self, action):
        self.current_state = torch.cat((self.current_state, torch.tensor([action])))

        generated_output = self.tokenizer.decode(self.current_state, skip_special_tokens=True).strip()
        expected_output = self.tokenizer.decode(self.current_sample['output_ids'], skip_special_tokens=True).strip()

        reward = evaluate(generated_output, expected_output)

        done = (len(self.current_state) >= self.max_length) or (action == self.tokenizer.eos_token_id)
        return self.current_state, reward, done, {}
