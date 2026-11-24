from wmx.train import baselines, crafter, frozenlake, sokoban, vector

RUNNERS = {"frozenlake": frozenlake.run, "crafter": crafter.run, "vector": vector.run, "sac": baselines.run_sac,
           "ppo": baselines.run_ppo, "iql": baselines.run_iql, "sokoban": sokoban.run}
