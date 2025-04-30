# decision_tree_driver.py
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""
import copy
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableSMC
import streamlit as st
import os


MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def save_tree_to_json(tree, leaf_probs, accuracy, tree_id):
    nodes_json = []
    leaf_set = set(int(leaf) for leaf in tree.leafs)
    leaf_id_to_index = {int(leaf): idx for idx, leaf in enumerate(tree.leafs)}

    # Save internal nodes
    for node in tree.tree:
        node_id, left_id, right_id, feature, threshold, depth = node
        node_json = {
            "id": int(node_id),
            "left": int(left_id),
            "right": int(right_id),
            "feature": int(feature),
            "threshold": float(threshold),
            "depth": int(depth),
            "is_leaf": False
        }
        nodes_json.append(node_json)

    # Save leaf nodes with probabilities
    for leaf_id in tree.leafs:
        leaf_idx = tree.leafs.index(leaf_id)
        probs = leaf_probs[leaf_id_to_index[int(leaf_id)]]
        probs_dict = {str(cls): float(prob) for cls, prob in probs.items()}
        leaf_json = {
            "id": int(leaf_id),
            "is_leaf": True,
            "probabilities": probs_dict
        }
        nodes_json.append(leaf_json)

    max_depth = max(node[5] for node in tree.tree)
    tree_data = {
        "nodes": nodes_json,
        "leafs": [int(leaf) for leaf in tree.leafs],
        "stats": {
            "num_nodes": len(tree.tree),
            "num_leaves": len(tree.leafs),
            "max_depth": int(max_depth),
            "accuracy": float(accuracy)
        }
    }
    with open(os.path.join(MODELS_DIR, f"tree_{tree_id}.json"), "w") as f:
        json.dump(tree_data, f, indent=4)

def train_smc_model(csv_path, target_column, tree_size, num_iterations, num_trees, resampling_scheme, random_state=42):
    """
    Trains SMC trees on the dataset at csv_path, using the specified target_column.
    Returns ensemble accuracy.
    Saves each tree as a JSON file (tree_0.json, tree_1.json, ...).
    """
    df = pd.read_csv(csv_path)
    df = df.dropna()
    # Separate target and features
    y = df[target_column].to_numpy()
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns].to_numpy()
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)
    # Then, after training and computing the SMC trees:
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    a = int(tree_size)
    target = dt.TreeTarget(a)
    initialProposal = dt.TreeInitialProposal(X_train, y_train)
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
    
    try:
        # Sample SMC trees
        treeSMCSamples, current_possibilities_for_predictions, logweights = dtSMC.sample(
            int(num_iterations), int(num_trees), a, resampling=resampling_scheme
        )
        # Predict using majority vote
        smcLabels, prob, leaf_prob, leaf_for_prediction = dt.stats(treeSMCSamples, X_test).predict(
            X_test, use_majority=True
        )
        smcAccuracy = dt.accuracy(y_test, smcLabels)
        print("SMC mean accuracy: ", smcAccuracy)
        
        # Also compute per-tree accuracies for saving
        smcLabel_per_tree = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=False)
        acc_per_tree = [dt.accuracy(y_test, label) for label in smcLabel_per_tree]
        
        # Save each tree to JSON
        for idx, (tree_sample, leaf_probs_sample, acc_val) in enumerate(zip(treeSMCSamples, current_possibilities_for_predictions, acc_per_tree)):
            save_tree_to_json(tree_sample, leaf_probs_sample, acc_val, idx)
            
            
        
        
        # ✅ Move feature importance saving HERE
        try:
            all_features = [col for col in df.columns if col != target_column]
            feature_usage = {i: 0 for i in range(len(all_features))}
            print("I am computing feature importance")
        
            for tree in treeSMCSamples:
                for node in tree.tree:
                    _, _, _, feature_idx, _, _ = node
                    if feature_idx >= 0:
                        feature_usage[feature_idx] += 1
        
            total_usage = sum(feature_usage.values())
            importance = {
                all_features[i]: (feature_usage[i] / total_usage) if total_usage > 0 else 0
                for i in feature_usage
            }
        
            with open("feature_importance.json", "w") as f:
                json.dump(importance, f, indent=2)
            print("Feature importance saved.")
        except Exception as e:
            print("Could not compute feature importance:", e)
        
        # ✅ THEN return
        return smcAccuracy
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
        return None
    
   

# For command-line usage, you could add:
if __name__ == "__main__":
    # Example usage: python decision_tree_driver.py datasets/LiverDisorder.csv Target 10 10 5 systematic
    if len(sys.argv) >= 7:
        csv_path = sys.argv[1]
        target_column = sys.argv[2]
        tree_size = sys.argv[3]
        num_iterations = sys.argv[4]
        num_trees = sys.argv[5]
        resampling_scheme = sys.argv[6]
        accuracy = train_smc_model(csv_path, target_column, tree_size, num_iterations, num_trees, resampling_scheme)
        if accuracy is not None:
            print("Overall ensemble accuracy:", accuracy)
    else:
        print("Usage: python decision_tree_driver.py <csv_path> <target_column> <tree_size> <num_iterations> <num_trees> <resampling_scheme>")
