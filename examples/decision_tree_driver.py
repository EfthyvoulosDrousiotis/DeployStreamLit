# decision_tree_driver.py
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021
@author: efthi
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableSMC
import streamlit as st

# ------------------------------------------------------------------#
#  Constants & helpers
# ------------------------------------------------------------------#
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------#
#  JSON saving utilities
# ------------------------------------------------------------------#
def save_tree_to_json(tree, leaf_probs, accuracy, tree_id):
    """Serialize one tree into models/tree_{id}.json"""
    nodes_json = []

    # internal nodes
    for node in tree.tree:
        node_id, left_id, right_id, feature, threshold, depth = node
        nodes_json.append(
            {
                "id": int(node_id),
                "left": int(left_id),
                "right": int(right_id),
                "feature": int(feature),
                "threshold": float(threshold),
                "depth": int(depth),
                "is_leaf": False,
            }
        )

    # leaf nodes
    leaf_id_to_index = {int(leaf): idx for idx, leaf in enumerate(tree.leafs)}
    for leaf_id in tree.leafs:
        probs = leaf_probs[leaf_id_to_index[int(leaf_id)]]
        probs_dict = {str(cls): float(prob) for cls, prob in probs.items()}
        nodes_json.append(
            {"id": int(leaf_id), "is_leaf": True, "probabilities": probs_dict}
        )

    max_depth = max(node[5] for node in tree.tree)
    tree_data = {
        "nodes": nodes_json,
        "leafs": [int(l) for l in tree.leafs],
        "stats": {
            "num_nodes": len(tree.tree),
            "num_leaves": len(tree.leafs),
            "max_depth": int(max_depth),
            "accuracy": float(accuracy) / 100 if accuracy > 1 else float(accuracy),
        },
    }

    with open(os.path.join(MODELS_DIR, f"tree_{tree_id}.json"), "w") as f:
        json.dump(tree_data, f, indent=4)

# ------------------------------------------------------------------#
#  NEW: row-weighted feature-importance
# ------------------------------------------------------------------#
def feature_importance_by_data(trees, X_train, verbose=False):
    """
    Row-weighted feature importance.
    Skips any child IDs not present in `tree.tree`.
    """
    n_samples, _ = X_train.shape
    counts = defaultdict(int)

    for t_idx, tree in enumerate(trees):
        # map id -> node tuple
        nodes = {n[0]: n for n in tree.tree}

        try:
            root_id = next(n_id for n_id, *rest in tree.tree if rest[-1] == 0)
        except StopIteration:
            if verbose:
                print(f"⚠️ Tree {t_idx}: no root (depth == 0). Skipping.")
            continue

        stack = [(root_id, np.ones(n_samples, dtype=bool))]

        while stack:
            node_id, mask = stack.pop()
            node = nodes.get(node_id)

            # orphan check
            if node is None:
                if verbose:
                    print(f"⚠️ Tree {t_idx}: orphan node id {node_id}.")
                continue

            _, left_id, right_id, feat_idx, thresh, _ = node
            if feat_idx < 0:  # leaf
                continue

            counts[feat_idx] += mask.sum()
            col_vals = X_train[:, feat_idx]

            left_mask  = mask & (col_vals <= thresh)
            right_mask = mask & (col_vals >  thresh)

            # Only push children that exist **and** receive rows
            if left_mask.any() and left_id in nodes:
                stack.append((left_id, left_mask))
            if right_mask.any() and right_id in nodes:
                stack.append((right_id, right_mask))

    total = sum(counts.values()) or 1
    return {f: c / total for f, c in counts.items()}


# ------------------------------------------------------------------#
#  Training driver
# ------------------------------------------------------------------#
def train_smc_model(
    csv_path,
    target_column,
    tree_size,
    num_iterations,
    num_trees,
    resampling_scheme,
    random_state=42,
):
    """
    Train an SMC ensemble and save:
      • tree_*.json in models/
      • feature_importance_split.json
      • feature_importance_rows.json
    Returns mean ensemble accuracy.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # ---- split X / y
    y = df[target_column].to_numpy()
    feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )

    # expose test data to Streamlit
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    a = int(tree_size)
    target = dt.TreeTarget(a)
    initialProposal = dt.TreeInitialProposal(X_train, y_train)
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)

    try:
        tree_samples, leaf_probs_list, _ = dtSMC.sample(
            int(num_iterations), int(num_trees), a, resampling=resampling_scheme
        )

        # majority-vote ensemble accuracy
        smcLabels, _, _, _ = dt.stats(tree_samples, X_test).predict(
            X_test, use_majority=True
        )
        ensemble_acc = dt.accuracy(y_test, smcLabels)
        print("SMC ensemble accuracy:", ensemble_acc)

        # per-tree accuracy + save JSONs
        per_tree_labels = dt.stats(tree_samples, X_test).predict(
            X_test, use_majority=False
        )
        per_tree_acc = [
            dt.accuracy(y_test, lbl) for lbl in per_tree_labels
        ]
        for idx, (tree, lp, acc) in enumerate(
            zip(tree_samples, leaf_probs_list, per_tree_acc)
        ):
            save_tree_to_json(tree, lp, acc, idx)

        # ------------------------------------------------------------------
        # Compute BOTH importance metrics
        # ------------------------------------------------------------------
        # A) Split-count (how many times feature appears)
        split_counts = defaultdict(int)
        for tree in tree_samples:
            for node in tree.tree:
                feat_idx = node[3]
                if feat_idx >= 0:
                    split_counts[feat_idx] += 1
        total_splits = sum(split_counts.values()) or 1
        importance_split = {
            feature_columns[i]: split_counts.get(i, 0) / total_splits
            for i in range(len(feature_columns))
        }
        with open("feature_importance_split.json", "w") as f:
            json.dump(importance_split, f, indent=2)

        # B) Row-weighted
        importance_rows_raw = feature_importance_by_data(tree_samples, X_train, verbose=True)
        importance_rows = {
            feature_columns[i]: importance_rows_raw.get(i, 0.0)
            for i in range(len(feature_columns))
        }
        with open("feature_importance_rows.json", "w") as f:
            json.dump(importance_rows, f, indent=2)

        print("Feature-importance files written.")

        return ensemble_acc / 100 if ensemble_acc > 1 else ensemble_acc

    except ZeroDivisionError:
        print("SMC sampling failed (division by zero)")
        return None

# ------------------------------------------------------------------#
#  CLI entry-point
# ------------------------------------------------------------------#
if __name__ == "__main__":
    # Usage: python decision_tree_driver.py <csv> <target> <a> <iters> <trees> <resampling>
    if len(sys.argv) >= 7:
        _, csv_path, target_column, a, iters, trees, scheme = sys.argv[:7]
        acc = train_smc_model(csv_path, target_column, a, iters, trees, scheme)
        if acc is not None:
            print("Overall ensemble accuracy:", acc)
    else:
        print(
            "Usage: python decision_tree_driver.py "
            "<csv_path> <target_column> <tree_size> <num_iterations> "
            "<num_trees> <resampling_scheme>"
        )
