import pandas as pd
from tqdm import tqdm
import gadfly
import numpy as np
import matplotlib.pyplot as plt
import scipy
import shap


def find_subtopics(io_pairs, topic):
    subtopics = {}
    for id,io_pair in io_pairs.iterrows():
        if io_pair.topic.startswith(topic):
            parts = io_pair.topic[len(topic):].split("/")
            if len(parts) > 1:
                subtopics[topic + "/" + parts[1]] = True
            else:
                subtopics[id] = True
    return list(subtopics.keys())

def filter_topics(io_pairs, subtopics):
    ids_to_keep = []
    for id,io_pair in io_pairs.iterrows():
        for topic in subtopics:
            if io_pair.topic.startswith(topic) or id == topic:
                ids_to_keep.append(id)
                break

    return io_pairs.filter(ids_to_keep, axis=0)

def find_intersection_and_disjunction(io_pairs1, io_pairs2):
    # io_pairs1 is suggestions, io_pairs2 is reference group
    reference_pairs = {}
    make_key = lambda io_pair: (io_pair.input.lower(), io_pair.output.lower())
    for id, io_pair in io_pairs2.iterrows():
        key = make_key(io_pair)
        reference_pairs[key] = id

    intersect_ids = []
    disjoint_ids = []
    for id, io_pair in io_pairs1.iterrows():
        key = make_key(io_pair)
        if key in reference_pairs:
            intersect_ids.append(reference_pairs[key])
        else:
            disjoint_ids.append(id)
    return io_pairs2.filter(intersect_ids, axis=0), io_pairs1.filter(disjoint_ids, axis=0)

def test_randomization(explorer, topic, io_pairs, randomizations, n_reps, max_generate):
    overlaps = []
    disjunctions = []

    for r in tqdm(randomizations):
        rep_results = []
        dis_results = []
        for i in range(n_reps):
            suggestions = explorer._generate_suggestions(
                topic, max_generate=max_generate, randomization=r
            )
            o, d = find_intersection_and_disjunction(suggestions, io_pairs)
            rep_results.append(o)
            dis_results.append(d)
        overlaps.append(rep_results)
        disjunctions.append(dis_results)
    return overlaps, disjunctions

def overlap_parameter_scan(generate_suggestions, io_pairs, params, n_reps):
    overlaps = []
    disjunctions = []

    for param in tqdm(params):
        rep_results = []
        dis_results = []
        for i in range(n_reps):
            suggestions = generate_suggestions(param)
            o, d = find_intersection_and_disjunction(suggestions, io_pairs)
            rep_results.append(o)
            dis_results.append(d)
        overlaps.append(rep_results)
        disjunctions.append(dis_results)
    return overlaps, disjunctions

def len_metric(overlaps, xs):
    raw_x = []
    raw_y = []
    mean_x = []
    mean_y = []
    std_x = []
    std_y = []
    for i in range(len(overlaps)):
        vals = [len(rep) for rep in overlaps[i]]
        for v in vals:
            raw_x.append(xs[i])
            raw_y.append(v)
        mean_x.append(xs[i])
        mean_y.append(np.mean(vals))
        std_y.append(np.std(vals))
    return [np.array(v) for v in [raw_x, raw_y, mean_x, mean_y, std_y]]

def plot_len_metric(overlaps, xs, score_threshold=0, xlabel=None, scatter=True):
    overlaps = [[v[v["score"] >= score_threshold] for v in o] for o in overlaps]
    raw_x, raw_y, mean_x, mean_y, std_y = len_metric(overlaps, xs)
    n_reps = len(overlaps[0])
    if scatter:
        plt.scatter(raw_x, raw_y, alpha=0.5, color=shap.plots.colors.blue_rgb)

    plt.fill_between(
        mean_x,
        mean_y + std_y/np.sqrt(n_reps),
        mean_y - std_y/np.sqrt(n_reps),
        alpha=0.2,
        color=shap.plots.colors.blue_rgb
    )
    plt.fill_between(
        mean_x,
        mean_y + 2*std_y/np.sqrt(n_reps),
        mean_y - 2*std_y/np.sqrt(n_reps),
        alpha=0.2,
        color=shap.plots.colors.blue_rgb
    )
    plt.plot(mean_x, mean_y, color=shap.plots.colors.blue_rgb)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel("# overlaps")
    plt.show()
