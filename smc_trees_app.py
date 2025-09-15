import streamlit as st
import json
import graphviz
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Import the training function from decision_tree_driver.py
from examples.decision_tree_driver import train_smc_model, save_tree_to_json

st.title("🌳 Sequential Monte Carlo Trees Dashboard")

# ----------------------
# Helper / constants
# ----------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def load_tree(tree_id):
    file_path = os.path.join(MODELS_DIR, f"tree_{tree_id}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as file:
        return json.load(file)

@st.cache_data(show_spinner=False)
def load_feature_names():
    with open("feature_names.json", "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def get_valid_tree_files():
    valid_files = []
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("tree_") and filename.endswith(".json"):
            try:
                with open(os.path.join(MODELS_DIR, filename), "r") as file:
                    data = json.load(file)
                if "nodes" in data and isinstance(data["nodes"], list):
                    valid_files.append(filename)
            except Exception as e:
                st.write(f"Skipping {filename} due to error: {e}")
    return valid_files

def encode_categoricals(df: pd.DataFrame, save_path: str = "categorical_encodings.json"):
    """
    • Converts object columns that are fully numeric (even if stored as str)
      → numeric dtype (Int64 or float64).
    • Converts remaining non-numeric object columns → categorical integer codes.
    • Returns (encoded_df, mapping_dict).  Mapping only for categorical columns.
    """
    enc_map  = {}
    df_enc   = df.copy()

    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col]):
            continue

        ser = (df[col]
               .astype(str)
               .str.strip()
               .replace({"": np.nan, "nan": np.nan, "?": np.nan}))

        numeric_try = pd.to_numeric(ser, errors="coerce")
        all_numeric = numeric_try.notna().sum() == ser.notna().sum()

        if all_numeric:
            # if all valid entries are whole numbers -> Int64
            vals = numeric_try.dropna()
            if len(vals) and (vals % 1 == 0).all():
                df_enc[col] = numeric_try.astype("Int64")
            else:
                df_enc[col] = numeric_try.astype(float)
        else:
            # real categorical
            codes, labels = pd.factorize(ser, sort=True)
            codes_ser = pd.Series(codes).replace({-1: pd.NA})
            df_enc[col] = codes_ser.astype("Int64")
            enc_map[col] = {str(label): int(code) for code, label in enumerate(labels)}

    if enc_map:
        Path(save_path).write_text(json.dumps(enc_map, indent=2))

    return df_enc, enc_map

def visualize_tree(tree_data, feature_names):
    dot = graphviz.Digraph()
    dot.attr('node', style='rounded,filled')

    for node in tree_data["nodes"]:
        if node["is_leaf"]:
            probs = node.get("probabilities", {})
            prob_str = "\n".join([f"Class {cls}: {prob*100:.1f}%" for cls, prob in probs.items()])
            label = f"Leaf {node['id']}\n{prob_str}"
            dot.node(str(node["id"]), label, shape='box',
                     style='filled,rounded', fillcolor='lightgreen', color='darkgreen')
        else:
            feature_idx = node['feature']
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
            label = f"{feature_name} ≤ {node['threshold']:.2f}"
            dot.node(str(node["id"]), label, shape='ellipse',
                     style='filled', fillcolor='lightblue', color='steelblue')

    for node in tree_data["nodes"]:
        if not node["is_leaf"]:
            dot.edge(str(node["id"]), str(node["left"]),  label="True")
            dot.edge(str(node["id"]), str(node["right"]), label="False")
    return dot

def predict_from_tree(tree, input_features):
    """
    Walks the tree for a single input and returns (probabilities, path).
    """
    nodes = {node["id"]: node for node in tree["nodes"]}

    root = next(
        (n for n in tree["nodes"]
         if not n.get("is_leaf", False) and n.get("depth", -1) == 0),
        None
    )
    if root is None:
        st.error("❌ No root node found in the tree JSON.")
        return {}, []

    path = [root["id"]]
    current = root

    while not current.get("is_leaf", False):
        if "feature" not in current or "threshold" not in current:
            st.error(f"❌ Node {current.get('id')} is missing 'feature' or 'threshold'.")
            return {}, path

        try:
            feature_idx = int(current["feature"])
            threshold = float(current["threshold"])
        except (TypeError, ValueError):
            st.error(f"❌ Invalid 'feature' or 'threshold' at node {current['id']}.")
            return {}, path

        if feature_idx < 0 or feature_idx >= len(input_features):
            st.error(f"❌ Feature index {feature_idx} out of range (have {len(input_features)} features).")
            return {}, path

        raw_val = input_features[feature_idx]
        try:
            feature_value = float(raw_val)
        except (TypeError, ValueError):
            st.error(f"❌ Invalid feature value for index {feature_idx}: {raw_val}")
            return {}, path

        next_id = current["left"] if feature_value <= threshold else current["right"]
        path.append(next_id)
        current = nodes.get(next_id)
        if current is None:
            st.error(f"❌ Could not find node with id {next_id} in tree JSON.")
            return {}, path

    return current.get("probabilities", {}), path

def visualize_tree_with_path(tree_data, feature_names, path):
    dot = graphviz.Digraph()
    dot.attr('node', style='rounded,filled')
    in_path = set(path)

    for node in tree_data["nodes"]:
        if node.get("is_leaf", False):
            probs = node.get("probabilities", {})
            prob_str = "\n".join([f"Class {cls}: {prob*100:.1f}%" for cls, prob in probs.items()])
            label = f"Leaf {node['id']}\n{prob_str}"
            if node["id"] in in_path:
                dot.node(str(node["id"]), label, shape='box',
                         style='filled,rounded', fillcolor='salmon', color='red')
            else:
                dot.node(str(node["id"]), label, shape='box',
                         style='filled,rounded', fillcolor='lightgreen', color='darkgreen')
        else:
            feature_idx = node["feature"]
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
            label = f"{feature_name} ≤ {node['threshold']:.2f}"
            if node["id"] in in_path:
                dot.node(str(node["id"]), label, shape='ellipse',
                         style='filled', fillcolor='lightsalmon', color='red')
            else:
                dot.node(str(node["id"]), label, shape='ellipse',
                         style='filled', fillcolor='lightblue', color='steelblue')

    for node in tree_data["nodes"]:
        if not node.get("is_leaf", False):
            left_id = node["left"]; right_id = node["right"]
            if node["id"] in in_path and left_id in in_path:
                dot.edge(str(node["id"]), str(left_id), label="True", color="red", penwidth="2")
            else:
                dot.edge(str(node["id"]), str(left_id), label="True")
            if node["id"] in in_path and right_id in in_path:
                dot.edge(str(node["id"]), str(right_id), label="False", color="red", penwidth="2")
            else:
                dot.edge(str(node["id"]), str(right_id), label="False")
    return dot

def build_label_to_tree_id():
    tree_files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith("tree_") and f.endswith(".json")])
    mapping = {}
    for filename in tree_files:
        try:
            tree_id = int(filename.split("_")[1].split(".")[0])
        except ValueError:
            continue
        data = load_tree(tree_id)
        if data and "stats" in data:
            stats = data["stats"]
            label = (f"Tree {tree_id} | Nodes: {stats['num_nodes']} | Leaves: {stats['num_leaves']} | "
                     f"Depth: {stats['max_depth']} | Accuracy: {stats['accuracy']:.2%}")
            mapping[label] = tree_id
    return mapping

from pathlib import Path

def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace({'': np.nan, '?': np.nan})
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].notna()
            if converted[mask].notna().all():
                non_na = converted.dropna()
                if len(non_na) and (non_na % 1 == 0).all():
                    df[col] = converted.astype("Int64")
                else:
                    df[col] = converted
    return df

# ----------------------
# Create Tabs
# ----------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🌲💪 Train SMC Model",
    "🌲 Single Tree View",
    "🌲🌳 Compare Trees",
    "📊 Feature Importance",
    "🎯 Interactive Prediction",
    "📈 Overall Performance",
    "🔒 Robustness Analysis",
    "📊 Statistical Tests",
    "📈 Custom Plotting",
    "Evaluation Metrics"
])

# ----------------------
# Tab 0: Train SMC Model
# ----------------------
with tab0:
    st.header("💪 Train SMC Trees Model")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="train_csv")

    if uploaded_file is not None:
        # Read as string → then clean/encode
        df_raw = pd.read_csv(uploaded_file, dtype=str)
        st.write("Dataset preview (first 10 rows):")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("Missing value summary")
        missing_summary = df_raw.replace({"": np.nan}).isna().sum()
        st.dataframe(missing_summary.to_frame("Missing"), use_container_width=True)

        if missing_summary.sum() > 0:
            with st.expander("🧹 Handle missing values"):
                choice = st.selectbox(
                    "Choose strategy",
                    ["Drop rows", "Fill with column mean", "Fill with column median", "Fill with specific value", "Leave as-is"],
                )
                if choice == "Drop rows":
                    df_clean = df_raw.replace({"": np.nan}).dropna()
                elif choice == "Fill with column mean":
                    df_tmp = df_raw.replace({"": np.nan})
                    df_tmp = df_tmp.apply(pd.to_numeric, errors="ignore")
                    df_clean = df_tmp.fillna(df_tmp.mean(numeric_only=True))
                elif choice == "Fill with column median":
                    df_tmp = df_raw.replace({"": np.nan})
                    df_tmp = df_tmp.apply(pd.to_numeric, errors="ignore")
                    df_clean = df_tmp.fillna(df_tmp.median(numeric_only=True))
                elif choice == "Fill with specific value":
                    val = st.number_input("Value to fill", value=0.0)
                    df_clean = df_raw.replace({"": np.nan}).fillna(val)
                else:
                    df_clean = df_raw.copy()
        else:
            df_clean = df_raw.copy()

        # Auto-encode categoricals / numeric-looking strings
        df_clean, enc_map = encode_categoricals(df_clean)
        if enc_map:
            st.success("Categorical columns auto-encoded.")
            enc_df = (
                pd.DataFrame(
                    [(col, orig, code) for col, m in enc_map.items() for orig, code in m.items()],
                    columns=["Feature", "Original label", "Encoded as"],
                )
                .sort_values(["Feature", "Encoded as"])
            )
            with st.expander("🔑 Encoding map"):
                st.dataframe(enc_df, use_container_width=True)
        else:
            st.info("No categorical columns detected.")

        all_cols = list(df_clean.columns)
        target_column = st.selectbox("Select target column", all_cols)
        feature_columns = [c for c in all_cols if c != target_column]
        st.write("**Features used:**", feature_columns)

        with open("feature_names.json", "w") as f:
            json.dump(feature_columns, f, indent=2)

        st.subheader("SMC parameters")
        tree_size       = st.number_input("Tree size (a)",          min_value=1, value=10, step=1)
        num_iterations  = st.number_input("Number of iterations",   min_value=1, value=10, step=1)
        num_trees       = st.number_input("Number of trees",        min_value=1, value=5,  step=1)
        resampling_opts = ["residual", "systematic", "knapsack", "min_error", "variational", "min_error_imp", "CIR"]
        resampling_scheme = st.selectbox("Resampling scheme", resampling_opts)

        csv_path = f"datasets/{uploaded_file.name}"
        os.makedirs("datasets", exist_ok=True)
        df_clean.to_csv(csv_path, index=False)

        if st.button("🚀 Train SMC Model"):
            for f in os.listdir(MODELS_DIR):
                if f.startswith("tree_") and f.endswith(".json"):
                    os.remove(os.path.join(MODELS_DIR, f))

            with st.spinner("Training, please wait…"):
                accuracy = train_smc_model(
                    csv_path,
                    target_column,
                    tree_size,
                    num_iterations,
                    num_trees,
                    resampling_scheme,
                )

            if accuracy is not None:
                st.success(f"Training done. Ensemble accuracy: {accuracy:.2%}")
                # Prepare a clean test split for downstream tabs
                X = df_clean[feature_columns].to_numpy()
                y = df_clean[target_column].to_numpy()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.30, random_state=42
                )
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.cache_data.clear()
            else:
                st.error("Training failed. Check the console logs.")

# ----------------------
# Tab 1: Single Tree View
# ----------------------
with tab1:
    st.header("Single Tree Visualization")
    tree_files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith("tree_") and f.endswith(".json")])

    if not tree_files:
        st.info("No trees found. Please train a model first (Tab 0).")
    else:
        label_to_tree_id = {}
        for filename in tree_files:
            try:
                tree_id = int(filename.split("_")[1].split(".")[0])
                data = load_tree(tree_id)
                if data and "stats" in data:
                    stats = data["stats"]
                    label = f"Tree {tree_id} | Nodes: {stats['num_nodes']} | Leaves: {stats['num_leaves']} | Depth: {stats['max_depth']} | Acc: {stats['accuracy']:.2%}"
                    label_to_tree_id[label] = tree_id
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")

        if label_to_tree_id:
            selected_label = st.selectbox("Select Tree:", list(label_to_tree_id.keys()))
            selected_tree_id = label_to_tree_id[selected_label]
            tree_data = load_tree(selected_tree_id)
            feature_names = load_feature_names()
            st.graphviz_chart(visualize_tree(tree_data, feature_names))
        else:
            st.info("No valid tree data available yet.")

# ----------------------
# Tab 2: Compare Trees (Side-by-Side)
# ----------------------
with tab2:
    st.header("Side-by-Side Tree Comparison")
    mapping = build_label_to_tree_id()
    if not mapping:
        st.error("No trees available. Please train the model first in the 'Train SMC Model' section.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            selected_label1 = st.selectbox("Select First Tree:", list(mapping.keys()), key='first')
            tree_id1 = mapping[selected_label1]
            tree_data1 = load_tree(tree_id1)
            stats1 = tree_data1["stats"]
            st.markdown(f"**Tree {tree_id1} Stats:**  \n- Nodes: {stats1['num_nodes']}  \n- Leaves: {stats1['num_leaves']}  \n- Depth: {stats1['max_depth']}  \n- Accuracy: {stats1['accuracy']:.2%}")
            tree_viz1 = visualize_tree(tree_data1, load_feature_names())
            st.graphviz_chart(tree_viz1)
        with col2:
            selected_label2 = st.selectbox("Select Second Tree:", list(mapping.keys()), index=1, key='second')
            tree_id2 = mapping[selected_label2]
            tree_data2 = load_tree(tree_id2)
            stats2 = tree_data2["stats"]
            st.markdown(f"**Tree {tree_id2} Stats:**  \n- Nodes: {stats2['num_nodes']}  \n- Leaves: {stats2['num_leaves']}  \n- Depth: {stats2['max_depth']}  \n- Accuracy: {stats2['accuracy']:.2%}")
            tree_viz2 = visualize_tree(tree_data2, load_feature_names())
            st.graphviz_chart(tree_viz2)

# ----------------------
# Tab 3 • Feature Importance + Consensus Tree
# ----------------------
from collections import defaultdict
import hashlib
import itertools

with tab3:
    st.header("📊 Feature Importance & Consensus Tree")

    metric_choice = st.radio(
        "Importance metric",
        ("Split frequency", "Rows handled"),
        horizontal=True,
        index=0,
    )
    file_map = {
        "Split frequency": "feature_importance_split.json",
        "Rows handled":    "feature_importance_rows.json",
    }
    imp_file = file_map[metric_choice]

    if not os.path.exists(imp_file):
        st.error(f"File '{imp_file}' not found. Train the model first.")
        st.stop()

    with open(imp_file, "r") as f:
        imp_dict = json.load(f)

    items = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    feats, imps = zip(*items) if items else ([], [])
    if len(imps) == 0:
        st.info("No feature-importance data found.")
    else:
        fig, ax = plt.subplots(figsize=(8, max(4, len(feats) * 0.4)))
        ax.barh(feats[::-1], [v * 100 for v in imps[::-1]])
        ax.set_xlabel("Importance (%)")
        ax.set_title(f"Feature importance • {metric_choice}")
        st.pyplot(fig)
        st.dataframe(
            {"Feature": feats, "Importance (%)": [round(v * 100, 2) for v in imps]},
            use_container_width=True,
        )

    if "X_test" not in st.session_state or "y_test" not in st.session_state:
        st.info("Train a model first to build the consensus tree.")
        st.stop()

    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    feature_names = load_feature_names()
    max_depth = st.slider("Consensus-tree max depth", 1, 6, 3)

    # --- Helpers robust to JSON schema/dtypes ---
    def _is_leaf(n):
        if n.get("is_leaf") is True or n.get("leaf") is True:
            return True
        l = n.get("left", n.get("left_id"))
        r = n.get("right", n.get("right_id"))
        return (l in (None, "", -1)) and (r in (None, "", -1))

    def _leaf_label(n):
        for k in ("class", "label", "pred", "prediction", "yhat", "y", "target", "class_index"):
            if k in n and n[k] is not None and not isinstance(n[k], (list, dict)):
                return str(n[k])
        for k in ("probabilities", "proba", "prob", "probs"):
            if k in n and n[k] is not None and isinstance(n[k], dict) and len(n[k]):
                return str(max(n[k].items(), key=lambda kv: kv[1])[0])
        for k in ("counts", "class_counts", "hist", "class_hist", "n_class", "counts_per_class"):
            if k in n and n[k] is not None and isinstance(n[k], dict) and len(n[k]):
                return str(max(n[k].items(), key=lambda kv: kv[1])[0])
        return "NA"

    def _feature_threshold(n):
        f = n.get("feature", n.get("feat", n.get("feature_index", n.get("split_feature"))))
        t = n.get("threshold", n.get("thr", n.get("split_threshold", n.get("value"))))
        if isinstance(t, (list, tuple, np.ndarray)):
            t = float(t[0])
        return int(f), float(t)

    def _node_map_and_root(nodes_list):
        nodes = {}
        child_ids = set()
        for n in nodes_list:
            nid = str(n.get("id", n.get("nid", n.get("node_id"))))
            nodes[nid] = n
            if not _is_leaf(n):
                l = n.get("left", n.get("left_id"))
                r = n.get("right", n.get("right_id"))
                if l is not None: child_ids.add(str(l))
                if r is not None: child_ids.add(str(r))
        for nid in nodes:
            if nid not in child_ids:
                return nodes, nid
        return nodes, next(iter(nodes))

    def _col_as_float(xcol):
        if np.issubdtype(xcol.dtype, np.number):
            return xcol.astype(float)
        return pd.to_numeric(xcol, errors="coerce").to_numpy()

    def predict_tree_json(tree, X):
        nodes, rid = _node_map_and_root(tree["nodes"])
        out = []
        for row in X:
            nid = rid
            while not _is_leaf(nodes[nid]):
                node = nodes[nid]
                f, thr = _feature_threshold(node)
                val = row[f]
                try:
                    v = float(val)
                except Exception:
                    v = np.nan
                if np.isnan(v):
                    # default to left if missing
                    nid = str(node.get("left"))
                else:
                    nid = str(node.get("left")) if v <= thr else str(node.get("right"))
            out.append(_leaf_label(nodes[nid]))
        return np.array(out, dtype=str)

    tree_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("tree_")]
    if len(tree_files) == 0:
        st.error("No trees found in the models directory.")
        st.stop()

    ensemble = [load_tree(int(f.split("_")[1].split(".")[0])) for f in tree_files]
    V = np.vstack([predict_tree_json(t, X_test) for t in ensemble])   # [T, N]
    T, N = V.shape
    w = np.ones(T, dtype=float)

    def per_row_majority(mask):
        cols = np.where(mask)[0]
        chosen = []
        for j in cols:
            vals, inv = np.unique(V[:, j], return_inverse=True)
            counts = np.bincount(inv, weights=w, minlength=len(vals))
            chosen.append(vals[int(np.argmax(counts))])
        return np.array(chosen, dtype=str)

    def region_majority(mask):
        cols = np.where(mask)[0]
        if len(cols) == 0:
            return "NA"
        label_counts = defaultdict(float)
        for j in cols:
            for i in range(T):
                label_counts[V[i, j]] += w[i]
        return max(label_counts.items(), key=lambda kv: kv[1])[0]

    def best_split(mask, depth_level):
        counts = defaultdict(int)
        if int(mask.sum()) == 0:
            return None
        for t in ensemble:
            nodes, rid = _node_map_and_root(t["nodes"])
            stack = [(rid, mask, 0)]
            while stack:
                nid, m, d = stack.pop()
                node = nodes[nid]
                if _is_leaf(node):
                    continue
                if d == depth_level:
                    f, thr = _feature_threshold(node)
                    counts[(f, float(thr))] += int(m.sum())
                    continue
                # propagate masks safely
                f, thr = _feature_threshold(node)
                col = _col_as_float(X_test[:, f])
                valid = m & ~np.isnan(col)
                if not valid.any():
                    continue
                lm = valid & (col <= thr)
                rm = valid & (col >  thr)
                if lm.any(): stack.append((str(node.get("left")),  lm, d+1))
                if rm.any(): stack.append((str(node.get("right")), rm, d+1))
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else None

    x_hash = hashlib.md5(X_test.tobytes()).hexdigest()
    y_hash = hashlib.md5(y_test.astype(str).tobytes()).hexdigest()
    tree_sig = (len(ensemble),) + tuple(sorted(tree_files))

    @st.cache_data(show_spinner=False)
    def build_consensus(depth_cap, x_sig, y_sig, tree_signature):
        def recurse(mask, depth):
            row_maj = per_row_majority(mask)
            uniq = np.unique(row_maj)
            if depth >= depth_cap or len(uniq) <= 1:
                pred = uniq[0] if len(uniq) == 1 else region_majority(mask)
                return {"leaf": True, "class": str(pred)}

            split = best_split(mask, depth)
            if split is None:
                pred = region_majority(mask)
                return {"leaf": True, "class": str(pred)}

            feat, thr = split
            col = _col_as_float(X_test[:, feat])
            valid = mask & ~np.isnan(col)
            if not valid.any():
                pred = region_majority(mask)
                return {"leaf": True, "class": str(pred)}

            left_mask  = valid & (col <= thr)
            right_mask = valid & (col >  thr)
            if not left_mask.any() or not right_mask.any():
                pred = region_majority(mask)
                return {"leaf": True, "class": str(pred)}

            return {
                "feature": int(feat),
                "threshold": float(thr),
                "left":  recurse(left_mask,  depth + 1),
                "right": recurse(right_mask, depth + 1),
            }

        tree_dict = recurse(np.ones(N, dtype=bool), 0)

        def predict_one(row, node):
            while not node.get("leaf", False):
                f = node["feature"]; thr = node["threshold"]
                try:
                    v = float(row[f])
                except Exception:
                    v = np.nan
                node = node["left"] if (not np.isnan(v) and v <= thr) else node["right"]
            return node["class"]

        preds = np.array([predict_one(r, tree_dict) for r in X_test], dtype=str)
        acc   = np.mean(preds == y_test.astype(str))

        def count_leaves(n):
            return 1 if n.get("leaf", False) else count_leaves(n["left"]) + count_leaves(n["right"])

        return tree_dict, acc, count_leaves(tree_dict)

    tree_dict, cons_acc, n_leaves = build_consensus(max_depth, x_hash, y_hash, tree_sig)

    # Proper Graphviz rendering of the whole consensus tree
    def render_consensus_graph(node, feature_names):
        dot = graphviz.Digraph()
        counter = itertools.count(0)

        def add(n):
            my_id = str(next(counter))
            if n.get("leaf", False):
                dot.node(my_id, f"class = {n['class']}", shape="oval", style="filled", fillcolor="lightgreen")
                return my_id
            lab = f"{feature_names[n['feature']]} ≤ {n['threshold']:.2f}"
            dot.node(my_id, lab, shape="ellipse", style="filled", fillcolor="lightblue")
            left_id = add(n["left"])
            right_id = add(n["right"])
            dot.edge(my_id, left_id, label="True")
            dot.edge(my_id, right_id, label="False")
            return my_id

        add(node)
        return dot

    st.graphviz_chart(render_consensus_graph(tree_dict, feature_names))
    st.success(f"Consensus-tree accuracy on test set: **{cons_acc:.2%}**")
    st.caption(f"{len(ensemble)} trees • depth cap {max_depth} • leaves {n_leaves}")

# ----------------------
# Tab 4: Interactive Prediction (Simplified & Improved Voting)
# ----------------------
with tab4:
    st.header("🎯 Interactive Prediction")

    all_feature_names = load_feature_names()
    feature_names_for_prediction = [name for name in all_feature_names if name.lower() != "target"]

    st.subheader("📝 Enter Input Values for Features")
    input_df = pd.DataFrame([[0.0] * len(feature_names_for_prediction)], columns=feature_names_for_prediction)
    edited_df = st.data_editor(input_df, use_container_width=True, key="input_table")
    input_values = edited_df.iloc[0].tolist()

    mode = st.radio("Prediction Mode", options=["Single Tree", "Ensemble"], key="pred_mode")

    if mode == "Single Tree":
        st.subheader("🌲 Select a Tree for Prediction")
        label_to_tree_id = build_label_to_tree_id()
        if not label_to_tree_id:
            st.warning("No trees found. Please train the model first.")
        else:
            selected_tree_label = st.selectbox("Select Tree", list(label_to_tree_id.keys()), key="pred_tree")
            selected_tree_id = label_to_tree_id[selected_tree_label]
            tree_data = load_tree(selected_tree_id)

            if st.button("🔍 Predict with Selected Tree"):
                pred_probs, path = predict_from_tree(tree_data, input_values)
                if pred_probs:
                    predicted_class = max(pred_probs, key=pred_probs.get)
                    confidence = pred_probs[predicted_class]
                    st.success(f"**Predicted Class:** `{predicted_class}`  \n**Confidence:** `{confidence:.2%}`")
                    st.subheader("Class Probabilities")
                    st.dataframe(pd.DataFrame.from_dict(pred_probs, orient="index", columns=["Probability (%)"]).applymap(lambda x: round(x * 100, 2)))
                    st.subheader("Tree Path Highlight")
                    st.graphviz_chart(visualize_tree_with_path(tree_data, all_feature_names, path))
                else:
                    st.error("Prediction failed for the selected tree.")

    elif mode == "Ensemble":
        if st.button("🔎 Predict with Ensemble"):
            tree_files = get_valid_tree_files()
            if not tree_files:
                st.warning("No trees found. Please train the model first.")
            else:
                vote_counter = {}
                for file in tree_files:
                    tree_id = int(file.split("_")[1].split(".")[0])
                    tree_data = load_tree(tree_id)
                    pred_probs, _ = predict_from_tree(tree_data, input_values)
                    if pred_probs:
                        top_class = max(pred_probs, key=pred_probs.get)
                        vote_counter[top_class] = vote_counter.get(top_class, 0) + 1

                total_votes = sum(vote_counter.values()) or 1
                vote_probabilities = {cls: count / total_votes for cls, count in vote_counter.items()}
                predicted_class = max(vote_probabilities.items(), key=lambda x: x[1])[0]

                st.success(f"**Ensemble Predicted Class:** `{predicted_class}`")
                st.subheader("Class Probabilities Based on Voting")
                st.dataframe(pd.DataFrame.from_dict(vote_probabilities, orient="index", columns=["Vote Share (%)"]).applymap(lambda x: round(x * 100, 2)))
                st.subheader("Raw Vote Count")
                st.json(vote_counter)

# ----------------------
# Tab 5: Overall Performance Analysis
# ----------------------
with tab5:
    st.header("Overall Performance Analysis")

    tree_files = get_valid_tree_files()
    if not tree_files:
        st.warning("No tree JSON files found. Please train your model first.")
    else:
        stats_list = []
        for filename in tree_files:
            try:
                with open(os.path.join(MODELS_DIR, filename), "r") as file:
                    data = json.load(file)
                stats = data["stats"]
                tree_id = int(filename.replace("tree_", "").replace(".json", ""))
                stats_list.append({
                    "Tree ID": tree_id,
                    "Accuracy": stats["accuracy"],
                    "Depth": stats["max_depth"],
                    "Nodes": stats["num_nodes"],
                    "Leaves": stats["num_leaves"]
                })
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")

        df_stats = pd.DataFrame(stats_list).sort_values("Tree ID")

        fig = px.scatter(
            df_stats,
            x="Nodes",
            y="Accuracy",
            hover_name="Tree ID",
            title="Tree Accuracy vs Tree Size",
            labels={"Nodes": "Tree Size (Number of Nodes)", "Accuracy": "Accuracy"}
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_id = st.selectbox("Select a Tree to Visualize", df_stats["Tree ID"].astype(str))
        selected_tree = load_tree(int(selected_id))
        feature_names = load_feature_names()
        st.graphviz_chart(visualize_tree(selected_tree, feature_names))

        if "X_test" in st.session_state and "y_test" in st.session_state:
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]

            ensemble_predictions = []
            for x in X_test:
                votes = []
                for tree_id in df_stats["Tree ID"]:
                    tree_data = load_tree(tree_id)
                    pred_probs, _ = predict_from_tree(tree_data, list(x))
                    if pred_probs:
                        pred_label = max(pred_probs.items(), key=lambda p: p[1])[0]
                        votes.append(pred_label)
                majority = max(set(votes), key=votes.count) if votes else "?"
                ensemble_predictions.append(majority)

            ensemble_accuracy = np.mean([str(p) == str(t) for p, t in zip(ensemble_predictions, y_test)])
        else:
            ensemble_accuracy = None

        if ensemble_accuracy is not None:
            st.markdown(f"""
            - **Ensemble (Majority Voting) Accuracy:** {ensemble_accuracy:.2%}  
            - **Average Max Depth:** {df_stats['Depth'].mean():.2f}  
            - **Average Number of Nodes:** {df_stats['Nodes'].mean():.2f}  
            - **Average Number of Leaves:** {df_stats['Leaves'].mean():.2f}  
            """)
        else:
            st.markdown(f"""
            - **Ensemble (Majority Voting) Accuracy:** Not available  
            - **Average Max Depth:** {df_stats['Depth'].mean():.2f}  
            - **Average Number of Nodes:** {df_stats['Nodes'].mean():.2f}  
            - **Average Number of Leaves:** {df_stats['Leaves'].mean():.2f}  
            """)

# ----------------------
# Tab 6: Robustness Analysis
# ----------------------
with tab6:
    st.header("Robustness Analysis")
    all_feature_names = load_feature_names()
    feature_names_for_prediction = [name for name in all_feature_names if name.lower() != "target"]
    st.subheader("Enter Base Feature Values")
    base_input_features = []
    for i, name in enumerate(feature_names_for_prediction):
        value = st.number_input(f"{name}:", value=0.0, key=f"robust_input_{i}")
        base_input_features.append(value)
    noise_level = st.slider("Noise Level (Standard Deviation)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    num_samples = st.slider("Number of Perturbations", min_value=1, max_value=100, value=20)
    mode = st.radio("Prediction Mode", options=["Single Tree", "Ensemble"], key="robust_mode")
    predictions_list = []
    label_to_tree_id = build_label_to_tree_id()
    st.write("Available trees:", list(label_to_tree_id.keys()))
    if mode == "Single Tree":
        selected_label = st.selectbox("Select Tree:", list(label_to_tree_id.keys()), key="robust_tree")
        tree_id = label_to_tree_id[selected_label]
        if tree_id is None:
            st.error(f"Selected label '{selected_label}' not found. Please retrain the model or refresh the mapping.")
        else:
            tree_data = load_tree(tree_id)
        for _ in range(num_samples):
            noise = np.random.normal(0, noise_level, size=len(base_input_features))
            perturbed_input = [base + n for base, n in zip(base_input_features, noise)]
            pred, _ = predict_from_tree(tree_data, perturbed_input)
            predictions_list.append(pred)
    elif mode == "Ensemble":
        all_files = get_valid_tree_files()
        for _ in range(num_samples):
            noise = np.random.normal(0, noise_level, size=len(base_input_features))
            perturbed_input = [base + n for base, n in zip(base_input_features, noise)]
            predictions = []
            for file in all_files:
                try:
                    tree_id = int(file.split("_")[1].split(".")[0])
                except ValueError:
                    continue
                tree_data = load_tree(tree_id)
                pred = predict_from_tree(tree_data, perturbed_input)[0]
                predictions.append(pred)
            avg_pred = {}
            for pred in predictions:
                for cls, prob in pred.items():
                    avg_pred[cls] = avg_pred.get(cls, 0) + prob
            if predictions:
                n = len(predictions)
                for cls in avg_pred:
                    avg_pred[cls] /= n
            predictions_list.append(avg_pred)
    all_classes = set()
    for pred in predictions_list:
        all_classes.update(pred.keys())
    results = {cls: [] for cls in all_classes}
    for pred in predictions_list:
        for cls in all_classes:
            results[cls].append(pred.get(cls, 0))
    if results:
        fig, ax = plt.subplots(figsize=(8, 6))
        data_to_plot = [results[cls] for cls in sorted(results.keys())]
        ax.boxplot(data_to_plot, labels=sorted(results.keys()))
        ax.set_xlabel("Class")
        ax.set_ylabel("Predicted Probability")
        ax.set_title("Distribution of Predicted Probabilities under Noise")
        st.pyplot(fig)
        st.subheader("Summary Statistics for Predictions")
        for cls in sorted(results.keys()):
            st.write(f"Class {cls}: Mean = {np.mean(results[cls]):.2f}, Std = {np.std(results[cls]):.2f}")
    else:
        st.write("No predictions generated.")

# ----------------------
# Tab 7: Statistical Tests
# ----------------------
with tab7:
    st.header("Statistical Tests")
    st.write("Upload a CSV file for running statistical tests on your data.")
    uploaded_stat_file = st.file_uploader("Choose a CSV file", type=["csv"], key="stat_tests_file")
    if uploaded_stat_file is not None:
        df_stat = pd.read_csv(uploaded_stat_file)
        df_stat = df_stat.dropna()
        st.write("Dataset Preview:")
        st.dataframe(df_stat.head(10))

        from scipy.stats import ttest_ind, ttest_rel, f_oneway
        # Try optional statsmodels (may not be available in all deployments)
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            sm_ok = True
        except Exception as e:
            sm_ok = False
            sm_err = e

        test_type = st.selectbox("Select Statistical Test", 
                                  ["Independent t-test", "Paired t-test", "One-way ANOVA", "ANCOVA"])

        if test_type == "Independent t-test":
            st.markdown("#### Independent t-test")
            st.write("Select a numeric variable and a grouping (categorical) variable with exactly 2 groups.")
            numeric_col = st.selectbox("Select Numeric Variable", df_stat.columns, key="ind_ttest_numeric")
            group_col = st.selectbox("Select Grouping Variable", df_stat.columns, key="ind_ttest_group")
            if st.button("Run Independent t-test", key="run_ind_ttest"):
                groups = df_stat[group_col].unique()
                if len(groups) != 2:
                    st.error("Independent t-test requires exactly 2 groups.")
                else:
                    data1 = df_stat[df_stat[group_col] == groups[0]][numeric_col].dropna()
                    data2 = df_stat[df_stat[group_col] == groups[1]][numeric_col].dropna()
                    stat, p = ttest_ind(data1, data2)
                    st.write(f"t-test statistic: {stat:.3f}, p-value: {p:.3f}")

        elif test_type == "Paired t-test":
            st.markdown("#### Paired t-test")
            st.write("Select two numeric columns that represent paired measurements.")
            col1 = st.selectbox("Select First Numeric Variable", df_stat.columns, key="paired_ttest_col1")
            col2 = st.selectbox("Select Second Numeric Variable", df_stat.columns, key="paired_ttest_col2")
            if st.button("Run Paired t-test", key="run_paired_ttest"):
                data1 = df_stat[col1].dropna()
                data2 = df_stat[col2].dropna()
                if len(data1) != len(data2):
                    st.error("The two columns must have the same number of observations for a paired t-test.")
                else:
                    stat, p = ttest_rel(data1, data2)
                    st.write(f"Paired t-test statistic: {stat:.3f}, p-value: {p:.3f}")

        elif test_type == "One-way ANOVA":
            st.markdown("#### One-way ANOVA")
            st.write("Select a numeric variable and a categorical grouping variable with 3 or more groups.")
            numeric_col = st.selectbox("Select Numeric Variable", df_stat.columns, key="anova_numeric")
            group_col = st.selectbox("Select Grouping Variable", df_stat.columns, key="anova_group")
            if st.button("Run One-way ANOVA", key="run_anova"):
                groups = df_stat[group_col].unique()
                if len(groups) < 3:
                    st.error("One-way ANOVA requires at least 3 groups.")
                else:
                    group_data = [df_stat[df_stat[group_col] == grp][numeric_col].dropna() for grp in groups]
                    stat, p = f_oneway(*group_data)
                    st.write(f"ANOVA F-statistic: {stat:.3f}, p-value: {p:.3f}")

        elif test_type == "ANCOVA":
            st.markdown("#### ANCOVA")
            if not sm_ok:
                st.warning(f"statsmodels not available for ANCOVA. ({sm_err})")
            else:
                st.write("Select a dependent (numeric) variable, a categorical factor, and a continuous covariate.")
                dep_var = st.selectbox("Select Dependent Variable", df_stat.columns, key="ancova_dep")
                factor = st.selectbox("Select Categorical Factor", df_stat.columns, key="ancova_factor")
                covariate = st.selectbox("Select Continuous Covariate", df_stat.columns, key="ancova_cov")
                if st.button("Run ANCOVA", key="run_ancova"):
                    formula = f"{dep_var} ~ C({factor}) + {covariate}"
                    model = smf.ols(formula, data=df_stat).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    st.write("ANCOVA results:")
                    st.dataframe(anova_table)

# ----------------------
# Tab 8: Custom Plotting
# ----------------------
with tab8:
    st.header("Custom Plotting")
    st.write("Upload a CSV file or use an existing one to plot your features.")

    uploaded_plot_file = st.file_uploader("Choose a CSV file for plotting", type=["csv"], key="plot_file")
    if uploaded_plot_file is not None:
        df_plot = pd.read_csv(uploaded_plot_file).dropna()
        st.dataframe(df_plot.head(10))

        plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram"], key="plot_type")

        if plot_type in ["Scatter Plot", "Line Plot", "Bar Plot"]:
            x_col = st.selectbox("Select X-axis Feature", df_plot.columns, key="plot_x")
            y_col = st.selectbox("Select Y-axis Feature", df_plot.columns, key="plot_y")
        elif plot_type == "Histogram":
            col = st.selectbox("Select Feature for Histogram", df_plot.columns, key="plot_hist")
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)

        fig, ax = plt.subplots()
        if plot_type == "Scatter Plot":
            ax.scatter(df_plot[x_col], df_plot[y_col])
            ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        elif plot_type == "Line Plot":
            ax.plot(df_plot[x_col], df_plot[y_col], marker="o", linestyle="-")
            ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"Line Plot: {x_col} vs {y_col}")
        elif plot_type == "Bar Plot":
            ax.bar(df_plot[x_col], df_plot[y_col])
            ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"Bar Plot: {x_col} vs {y_col}")
        elif plot_type == "Histogram":
            ax.hist(df_plot[col], bins=bins, edgecolor="black")
            ax.set_xlabel(col); ax.set_ylabel("Frequency"); ax.set_title(f"Histogram of {col}")

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please upload a CSV file to create plots.")

# ----------------------
# Tab 9: Evaluation Metrics (ROC, AUC & Confusion Matrix)
# ----------------------
with tab9:
    st.header("Evaluation Metrics (ROC, AUC & Confusion Matrix)")

    if "X_test" not in st.session_state or "y_test" not in st.session_state:
        st.error("Test data not found. Please train the model first (Tab 0).")
        st.stop()

    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    tree_files = get_valid_tree_files()
    label_to_tree_id_eval = {}
    for filename in sorted(tree_files):
        try:
            tid = int(filename.split("_")[1].split(".")[0])
            data = load_tree(tid)
            if not data:
                continue
            stats = data.get("stats", {})
            acc = stats.get("accuracy", None)
            acc_txt = f"{acc:.2%}" if isinstance(acc, (int, float)) else "n/a"
            label = f"Tree {tid} | Nodes: {stats.get('num_nodes','?')} | Depth: {stats.get('max_depth','?')} | Acc: {acc_txt}"
            label_to_tree_id_eval[label] = tid
        except Exception:
            continue

    if not label_to_tree_id_eval:
        st.error("No tree models found. Please train the model first in the 'Train SMC Model' tab.")
        st.stop()

    eval_mode = st.radio("Prediction Mode", options=["Single Tree", "Ensemble"], horizontal=True, key="eval_mode")

    y_true_str = y_test.astype(str)
    unique_classes = np.unique(y_true_str)
    is_binary = len(unique_classes) == 2
    pos_label_str = unique_classes[-1] if is_binary else None

    y_scores = []
    y_pred_labels_str = []

    def argmax_label(prob_dict: dict) -> str:
        return max(prob_dict.items(), key=lambda kv: kv[1])[0] if prob_dict else "?"

    if eval_mode == "Single Tree":
        selected_label_eval = st.selectbox("Select Tree for Evaluation", list(label_to_tree_id_eval.keys()), key="eval_tree")
        tree_id_eval = label_to_tree_id_eval[selected_label_eval]
        tree_data_eval = load_tree(tree_id_eval)

        for row in X_test:
            pred_probs, _ = predict_from_tree(tree_data_eval, list(row))
            y_pred_labels_str.append(argmax_label(pred_probs))
            if is_binary:
                y_scores.append(float(pred_probs.get(pos_label_str, 0.0)))

    else:  # Ensemble
        all_tree_ids = [label_to_tree_id_eval[k] for k in label_to_tree_id_eval.keys()]
        for row in X_test:
            per_tree_probs = []
            votes = []
            for tid in all_tree_ids:
                td = load_tree(tid)
                probs, _ = predict_from_tree(td, list(row))
                per_tree_probs.append(probs)
                votes.append(argmax_label(probs))
            majority = max(set(votes), key=votes.count) if votes else "?"
            y_pred_labels_str.append(majority)
            if is_binary:
                cls_probs = [float(p.get(pos_label_str, 0.0)) for p in per_tree_probs] if per_tree_probs else [0.0]
                y_scores.append(float(np.mean(cls_probs)))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    labels_all = sorted(np.unique(np.concatenate([unique_classes, np.array(y_pred_labels_str, dtype=str)])))
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true_str, np.array(y_pred_labels_str, dtype=str), labels=labels_all)

    normalize = st.checkbox("Normalize rows to percentage", value=False, help="Show each row as proportions of the true class total.")
    cm_to_show = cm.astype(float)
    if normalize:
        row_sums = cm_to_show.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_to_show = (cm_to_show / row_sums) * 100.0

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm_to_show, cmap="Blues")
    ax_cm.set_xticks(range(len(labels_all)))
    ax_cm.set_yticks(range(len(labels_all)))
    ax_cm.set_xticklabels(labels_all)
    ax_cm.set_yticklabels(labels_all)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title("Confusion Matrix" + (" (%)" if normalize else " (counts)"))
    for i in range(cm_to_show.shape[0]):
        for j in range(cm_to_show.shape[1]):
            val = cm_to_show[i, j]
            txt = f"{val:.1f}%" if normalize else f"{int(val)}"
            ax_cm.text(j, i, txt, ha="center", va="center", color="black")
    plt.tight_layout()
    st.pyplot(fig_cm)

    with st.expander("Show classification report"):
        try:
            report = classification_report(y_true_str, np.array(y_pred_labels_str, dtype=str), labels=labels_all, zero_division=0, output_dict=False)
            st.text(report)
        except Exception as e:
            st.write("Could not compute classification report:", e)

    # ROC & AUC (binary only)
    st.subheader("ROC & AUC")
    if not is_binary:
        st.info("ROC/AUC shown only for binary problems. Detected classes: " + ", ".join(map(str, unique_classes)))
    else:
        if not y_scores:
            st.warning("No probability scores available for ROC.")
        else:
            fpr, tpr, thresholds = roc_curve(y_true_str, np.array(y_scores, dtype=float), pos_label=pos_label_str)
            roc_auc = auc(fpr, tpr)
            st.write(f"ROC AUC: {roc_auc:.3f}")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
            ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            st.pyplot(fig)

