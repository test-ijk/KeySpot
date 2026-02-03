import copy
import logging
import supersuit as ss

from harl.envs.pettingzoo_mw.walker.multiwalker.mw_move import (
    VIEWPORT_W,
    FPS,
    SCALE,
)

# from pettingzoo.sisl import multiwalker_v9
from harl.envs.pettingzoo_mw.walker.multiwalker.multiwalker import env_with_raw
from harl.envs.pettingzoo_mw.walker.multiwalker.mw_move import MultiWalkerEnv
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from .llm.agent import generate_prompt
import os
from dotenv import load_dotenv

from openai import OpenAI

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def get_model_response(model, prompt_content):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_content}],
        )
        print(response)
        return model, response.choices[0].message.content
    except Exception as e:
        return model, f"Error: {str(e)}"


class PettingZooMWLLMEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 500
            self.args["max_cycles"] = 501
        self.cur_step = 0
        # self.module = multiwalker_v9
        self.base_env, self.raw_env = env_with_raw(**self.args)
        self.multiwalker_env: MultiWalkerEnv = self.raw_env.env  # type: ignore
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(aec_to_parallel_wrapper(self.base_env))
        )
        self._seed = 0
        self.env.reset(seed=self._seed)
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(self.env.state_space)  # type: ignore
        self.observation_space = self.unwrap(
            {agent: self.env.observation_space(agent) for agent in self.agents}
        )
        self.action_space = self.unwrap(
            {agent: self.env.action_space(agent) for agent in self.agents}
        )

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, rew, term, trunc, info = self.env.step(self.wrap(actions))  # type: ignore
        obs_unwrapped = self.unwrap(obs)

        self.cur_step += 1

        if self.cur_step % 50 == 0:  
            llm_obses = [obs_unwrapped[agent_id] for agent_id in range(self.n_agents)]
            lidar_obs = self.get_thru_lidar_obs()
            llm_lidar_obses = [lidar_obs[agent_id] for agent_id in range(self.n_agents)]
            target_vs = [
                self.multiwalker_env.get_target_v_agent(agent_id)
                for agent_id in range(self.n_agents)
            ]
            prompt = generate_prompt(llm_obses, llm_lidar_obses, target_vs)  # needs
            # start_time = time.time()
            model, response = get_model_response("gpt-4o", prompt)
            # rich.print(response)
            # end_time = time.time()
            # rich.print(f"Time taken: {end_time - start_time} seconds")
            if response is None:
                raise Exception("Response is None")
            import json

            target_vs = json.loads(response)["target_vs"]
            for agent_id in range(self.n_agents):
                self.multiwalker_env.set_t_v_agent(agent_id, target_vs[agent_id])
                print(f"Changed actor {agent_id}  target_v to {target_vs[agent_id]}")

        for agent in self.agents:
            assert self.multiwalker_env.package is not None
            assert self.multiwalker_env.walkers is not None
            assert self.multiwalker_env.walkers[0] is not None
            assert self.multiwalker_env.walkers[0].hull is not None
            info[agent]["package_angle"] = (
                self.multiwalker_env.package.angle / 3.14 * 180
            )
            info[agent]["curr_step"] = self.cur_step
            info[agent]["package_x"] = self.multiwalker_env.package.position.x
            info[agent]["v_deviation"] = abs(
                self.multiwalker_env.target_v
                - 0.3
                * self.multiwalker_env.walkers[0].hull.linearVelocity.x
                * (VIEWPORT_W / SCALE)
                / FPS
            )
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True

        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        s_obs = self.repeat(self.env.state())  # type: ignore
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def get_thru_lidar_obs(self):
        return self.multiwalker_env.get_thru_lidar_obs()

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        assert self.env is not None
        obs, infos = self.env.reset(seed=self._seed)  # type: ignore
        obs = self.unwrap(obs)
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        render_result = self.raw_env.render()
        return render_result

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed
        self.env.reset(seed=self._seed)

    def wrap(self, lam):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = lam[i]
        return d

    def unwrap(self, d):
        _tmp = []
        for agent in self.agents:
            _tmp.append(d[agent])
        return _tmp

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
