#!/usr/bin/env python
# encoding: utf-8
import configparser
import json
from flask import Flask, request, jsonify
from lightning_gym.graph_utils import *
from lightning_gym.utils import *
from lightning_gym.envs.lightning_network import NetworkEnvironment
from baselines import *
from ActorCritic import DiscreteActorCritic
from collections import defaultdict
from cachelib import SimpleCache

app = Flask(__name__)

CONFIG_LOC = "./configs/default_api.conf"


def convert_to_dict(config) -> defaultdict:
    config_dict = defaultdict(dict)
    for section in config.sections():
        for key, value in config.items(section):
            config_dict[section][key] = value

    return config_dict


class Cache(object):
    cache = SimpleCache(threshold=1000, default_timeout=1000)

    @classmethod
    def get(cls, key=None):
        return cls.cache.get(key)

    @classmethod
    def delete(cls, key=None):
        return cls.cache.delete(key)

    @classmethod
    def set(cls, key=None, value=None, timeout=0):
        if timeout:
            return cls.cache.set(key, value, timeout=timeout)
        else:
            return cls.cache.set(key, value)

    @classmethod
    def clear(cls):
        return cls.cache.clear()


@app.route('/', methods=['POST'])
def run_sim():
    results = Cache.get(request.data)
    if results is None:
        record = json.loads(request.data)
        print(record)

        config = configparser.ConfigParser(allow_no_value=True)
        config.read(CONFIG_LOC)
        config_dict = convert_to_dict(config)
        config_dict.update(record)
        config.read_dict(config_dict)
        print(convert_to_dict(config))

        seed = config["env"].getint("seed", fallback=None)
        if seed:
            random_seed(seed)
            print("seed set")

        g, k_to_a, _ = create_snapshot_env(config)
        env = NetworkEnvironment(config, g=g)
        agent_type = config.get("agent", "type")
        agent = None
        if agent_type == "a2c":
            agent = TrainedGreedyAgent(env, config, n=1)
        elif agent_type == "random":
            agent = RandomAgent(env)
        elif agent_type == "topk_btwn":
            agent = TopBtwnAgent(env)
        elif agent_type == "topk_degree":
            agent = TopDegreeAgent(env)
        elif agent_type == "kcenter":
            agent = kCenterAgent(env)
        elif agent_type == "trained":
            agent = TrainedGreedyAgent(env, config, n=3)
        results = {
            "betweenness": agent.run_episode(),
            "recommendations": {k_to_a[key]: key for key in agent.problem.get_recommendations()}
        }
        print(results)
        Cache.set(request.data, results)
    return jsonify(results)


app.run(debug=True, host='0.0.0.0')
