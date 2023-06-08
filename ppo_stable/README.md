#### stable PPO


      # TODO
        test - halfcheetah
        test - custom pendulum



```python

action = dist.sample() # grad x
old_log_prob = dist.log_prob(action) # grad o

# print(action, [action.item()])
# action : -0.9288
# [action.item()] : [-0.928832471370697]

next_state, reward, terminated, truncated, _ = self.env.step([action.item()])

```
