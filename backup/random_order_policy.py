from envs.serial_multi_stage import SerialMultiStageEnv
from wrappers.multi_agent_wrapper import MultiAgentSerialWrapper

env_base = SerialMultiStageEnv(
    n_stages=1,
    lead_times=[2],  #  the real lead time = input_value + 1
    episode_length=100,
    render_mode=None,
)

env = MultiAgentSerialWrapper(env_base)

obs, info = env.reset()
done = {aid: False for aid in env.agent_ids}
truncated = {aid: False for aid in env.agent_ids}

while not all(done.values()) and not all(truncated.values()):
    # can generate actions for each agent using different policies.
    action = {
        aid: env.action_space[aid].sample()
        for aid in env.agent_ids
    }

    obs, rewards, done, truncated, infos = env.step(action)
    print("rewards:", rewards)
