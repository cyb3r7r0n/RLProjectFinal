import copy
from arena import Arena
from agent import CNetAgent
from switch_game import SwitchGame
from switch_cnet import SwitchCNet

import matplotlib.pyplot as plt

opt = {
        "game": "switch",
        "game_nagents": 3,
        "game_action_space": 2,
        "game_comm_limited": True,
        "game_comm_bits": 1,
        "game_comm_sigma": 2,
        "nsteps": 10,
        "gamma": 1,
        "model_dial": False,
        "model_target": True,
        "model_bn": True,
        "model_know_share": True,
        "model_action_aware": True,
        "model_rnn_size": 128,
        "model_rnn_layers":2,
        "model_avg_q":True,
        "bs": 32,
        "learningrate": 0.0005,
        "momentum": 0.05,
        "eps": 0.05,
        "eps_decay":1.0,
        "nepisodes": 1001,
        "step_test": 500,
        "step_target": 100,
        "td_lambda":2,
        "cuda": 0
     }


def create_agents(opt, game):
    agents = [None]  # 1-index agents
    cnet = SwitchCNet(opt)
    cnet_target = copy.deepcopy(cnet)
    for i in range(1, opt["game_nagents"] + 1):
        agents.append(CNetAgent(opt, game=game, model=cnet, target=cnet_target, index=i))
        if not opt["model_know_share"]:
            cnet = SwitchCNet(opt)
            cnet_target = copy.deepcopy(cnet)
    return agents

def run_trial(opt):

    game = SwitchGame(opt)
    agents = create_agents(opt, game)
    arena = Arena(opt, game)
    episode_numbers, rewards, std_devs = arena.train(agents)

    plt.figure()
    plt.plot(episode_numbers, rewards, label = "Average Rewards")
    # plt.plot(episode_numbers, std_devs, label = "Standard Deviation")
    if opt["model_dial"] == True:
        plt.title("DIAL")
    else:
        plt.title("RIAL")   
    plt.ylabel("Average Reward")
    plt.xlabel("Num Episodes")
    if opt["model_dial"] == True:
        plt.savefig("results/dial.png")
    else:
        plt.savefig("results/rial.png")
    plt.close("all")

    ACTIONS = [None, 'The prisoner decided to do nothing.', 'The prisoner chose to tell.']

    game.reset()
    batch = 0
    
    for j in range(10):
      ep = arena.run_episode(agents, False)
      print("\n\n\nEpisode", j+1)
      for i, step in enumerate(ep.step_records[:-1]):
          print('Day', i + 1)
          active_agent = game.active_agent[batch][i].item()
          print('Prisoner Selected for the interrogation: ', active_agent)
          print(ACTIONS[step.a_t[batch].detach().numpy()[active_agent - 1]])
          if step.comm[batch].detach().numpy()[active_agent - 1][0] == 1.0:
              print('The prisoner toggled the light bulb.')
          print()
          if step.terminal[batch]:
              break


if __name__ == '__main__':

     opt["model_comm_narrow"] = opt["model_dial"]

     if not opt["model_comm_narrow"] and opt["game_comm_bits"] > 0:
        opt["game_comm_bits"] = 2**opt["game_comm_bits"]

     if opt["game_comm_bits"] > 0 and opt["game_nagents"] > 1:
        opt["comm_enabled"] = True
        opt["game_action_space_total"] = opt["game_action_space"] + opt["game_comm_bits"]
     else:
        opt["game_action_space_total"] = opt["game_action_space"]
        opt["comm_enabled"] = False

     run_trial(opt)
