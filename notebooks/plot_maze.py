import os
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
os.environ["ROOT_DIR"] = ".."  # this allows to use relative paths in config files
from result_inspection.toy_maze import *
from result_inspection.plot_helpers import *

plot_kwargs = dict(stat_list=['cumulative_rew'], labels=['Reward'], figsize=(6, 4), titlesize=14)
# skill_kwargs = dict(figsize=(5,5), reset_dict=dict(state=torch.tensor([0., -0.5])))
start_state = [[0., -0.5], [0, -2], [2, -0.5], [4, -0.5], [2, -2], [4, -2], [2, -4], [4, -4], [0, -4]]
start_state = torch.tensor(start_state)


algo = "endpoint_contrastive_mi_06091023"
# algo = "contrastive_mi_06052350"
env = "square_maze"

exp, cmap = load_exp_data(
    "{}/{}".format(env, algo), notebook_mode=False
)  # NOTE: load agent, eval stats and episodes
for i, s in enumerate(start_state):
    ax = plot_all_skills(exp, cmap, notebook_mode=False, reset_dict=dict(state=s))#**skill_kwargs)  # NOTE: sample 20 trajs for each skill
    save_path = "../MI-result/images/"+algo+"_"+env+'_'+str(i)+'.png'
    plt.savefig(save_path)
    plt.close()


# i = 0
# save_path = "../MI-result/images/"+algo+"_"+env+'_'+str(i)+'.png'
# while os.path.exists(save_path):
#     i+=1
#     save_path = "../MI-result/images/"+algo+"_"+env+'_'+str(i)+'.png'
#     plt.savefig(save_path)

# axes = exp.plot_all()
# plot_stats_by_name(exp)
# plt.savefig("../MI-result/images/"+algo+"_"+env+"_stats"+'.png')