import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")


def load_data():
    path = "Cannabis.csv"
    out = pd.read_csv(path)
    out = out.drop(1390)
    return out


tds_dir = "results/"
if not os.path.exists(tds_dir):
    os.makedirs(tds_dir)

cannabis = load_data()

# ------------------------------------------------------------------------------------------------------
#                                           Effects                                                     #
# ------------------------------------------------------------------------------------------------------


def effect_understanding(save=False):
    effects = cannabis.Effects.to_numpy()
    tot_num = len(effects)
    effects_splilt = [v.split(",") for v in effects]
    effects_split_array = [[q.replace("\n", "").replace(" ", "").replace("Energentic", "Energetic") for q in v] for v in effects_splilt]
    effects_npy = np.array([v.replace("\n", "").replace(" ", "").replace("Energentic", "Energetic") for q in effects_splilt for v in q])
    effects_unique, effects_count = np.unique(effects_npy, return_counts=True)
    effects_index = np.arange(len(effects_unique))
    effects_split_array_num = []
    effects_correlation = np.zeros([len(effects_unique), len(effects_unique)])
    for v in effects_split_array:
        ind = [np.where(effects_unique == q)[0][0] for q in v]
        for i_index, i in enumerate(ind):
            for j_index, j in enumerate(ind):
                if i != j:
                    effects_correlation[i, j] += 1
        effects_split_array_num.append(np.array(ind))

    sort_index = np.argsort(effects_count)[::-1]
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121)
    x_axis = np.arange(len(effects_unique)) * 2
    ax.bar(x_axis, effects_count[sort_index] / tot_num, width=0.8, color='r')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(effects_unique[sort_index], rotation=45, fontsize=8)
    ax.set_xlabel("Effects", fontsize=8)
    ax.set_ylabel("Percentage out of %d cannabis" % tot_num, fontsize=8)
    ax.set_title("Distribution of the effects", fontsize=8)
    ax = fig.add_subplot(122)
    annotation = np.array(["%d" % v if v > 0 else "" for q in effects_correlation for v in q])
    annotation = np.reshape(annotation, effects_correlation.shape)
    output = sns.heatmap(effects_correlation, annot=annotation, ax=ax, cbar=False, cmap='YlGnBu', fmt='s',
                         xticklabels=effects_unique, yticklabels=effects_unique)
    plt.xticks(fontsize=8, rotation=45)
    ax.set_title("Correlation between effects", fontsize=8)
    plt.subplots_adjust(wspace=0.2)
    if save:
        plt.savefig(tds_dir + "/effects_understanding.pdf", pad_inches=0, bbox_inches='tight')
    return effects_split_array, effects_split_array_num, effects_unique, effects_count, effects_correlation


c_type = cannabis.Type.to_numpy()
def get_effects_wrt_type(c_type, effects_split_array_num, effects_name, save=False):
    c_unique_type, c_type_count = np.unique(c_type, return_counts=True)
    num_effect = len(effects_name)
    effects_type = {}
    for i, s in enumerate(c_unique_type):
        index = np.where(c_type == s)[0]
        effects_subset = [q for v in index for q in effects_split_array_num[v]]
        effects_unique, effects_count = np.unique(effects_subset, return_counts=True)
        effect_array = np.zeros(num_effect)
        effect_array[effects_unique] = effects_count
        effects_type[s] = effect_array

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    x_axis = np.arange(num_effect) * 3
    color_group = ['r', 'g', 'b']
    width = 0.5
    for i, s_key in enumerate(effects_type.keys()):
        ax.bar(x_axis + i * width, effects_type[s_key], width=width, color=color_group[i])
    ax.set_xticks(x_axis + 1 * width)
    ax.set_xticklabels(effects_name, rotation=45, fontsize=8)
    ax.legend([v + ":%d" % q for v, q in zip(list(effects_type.keys()), c_type_count)], fontsize=8, loc='best')
    ax.set_ylabel("Count", fontsize=8)
    plt.subplots_adjust(bottom=0.2)
    if save:
        plt.savefig(tds_dir + "/type_effects_understanding.pdf", pad_inches=0, bbox_inches='tight')

    return effects_type


