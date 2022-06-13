from helpers import *
from main import create_snapshot_env, random_seed
from baselines import TrainedGreedyAgent
import configparser
import pandas as pd

# pd.set_option('display.max_colwidth', None)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_loc = "./configs/test_snapshot.conf"
    config.read(config_loc)
    seed = config["env"].getint("seed", fallback=None)
    # print_config(config)
    if seed:
        random_seed(seed)
    config["env"]["filename"] = "feb.json"
    config["env"]["repeat"] = "True"
    # load the graph from the snapshots dir
    # TODO TODO
    # TODO TODO
    # save the new graph to the cleaned dir

    env, _, (size_before, size_after) = create_snapshot_env(config)
    # save_graph(env.base_graph,"test.json")
    agent = TrainedGreedyAgent(env, config, n=1)
    agent.run_episode()
    nbr_ids = agent.problem.get_recommendations()

    node_id = "it's working"
    data_dir = "./snapshots"
    new_edges = make_edges_from_template(node_id, nbr_ids)
    new_node = make_node_from_template(node_id)
    directed_edges = preprocess_json_file(f"{data_dir}/feb.json", new_node, new_edges)
    merchant_data = get_merchant_data()
    merchant_keys = list(merchant_data.keys())

    # SIMULATOR PARAMS
    transaction_size = 60000
    num_transactions = 10000
    epsilon = 0.8
    drop_disabled = True
    drop_low_cap = False
    with_depletion = True

    simulator = ts.TransactionSimulator(directed_edges,
                                        merchant_keys,
                                        transaction_size,
                                        num_transactions,
                                        drop_disabled=drop_disabled,
                                        drop_low_cap=drop_low_cap,
                                        epsilon=epsilon,
                                        with_depletion=with_depletion
                                        )
    cheapest_paths, _, all_router_fees, _ = simulator.simulate(weight="total_fee",
                                                               max_threads=16,
                                                               with_node_removals=False)
    output_dir = "test"
    total_income, total_fee = simulator.export(output_dir)
    top_5_income = total_income.sort_values("fee", ascending=False).set_index("node").head(5)
    top_5_traffic = total_income.sort_values("num_trans", ascending=False).set_index("node").head(5)
    node_income = all_router_fees.groupby("node")["fee"].sum().get(node_id, 0)  # return 0 if node did not make any fees
    node_traffic = total_income.sort_values("num_trans", ascending=False).get(node_id, 0)
    # median = all_router_fees.groupby("node")["fee"].sum().median()
    print("Top 5 earners:")
    print(top_5_income)
    print("Top 5 traffic:")
    print(top_5_traffic)
    # print("Median")  # 50% of nodes make less than this amount per day
    # print(median)
    print("Our Income:")
    print(node_income)
    # print("Our Traffic:")
    # print(node_traffic)
