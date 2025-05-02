import streamlit as st
import json
import graphviz
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
# Import the training function from decision_tree_driver.py
from examples.decision_tree_driver import train_smc_model, save_tree_to_json

st.title("🌳 Sequential Monte Carlo Trees Dashboard")

# ----------------------
# Helper Functions
# ----------------------
# Define a dedicated folder for tree models
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

def visualize_tree(tree_data, feature_names):
    dot = graphviz.Digraph()
    for node in tree_data["nodes"]:
        if node["is_leaf"]:
            probs = node["probabilities"]
            prob_str = "\n".join([f"Class {cls}: {prob*100:.1f}%" for cls, prob in probs.items()])
            label = f"Leaf {node['id']}\n{prob_str}"
            dot.node(str(node["id"]), label, shape='box', style='filled', color='lightgreen')
        else:
            feature_idx = node['feature']
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
            label = f"{feature_name} ≤ {node['threshold']:.2f}"
            dot.node(str(node["id"]), label, shape='ellipse', style='filled', color='lightblue')
    for node in tree_data["nodes"]:
        if not node["is_leaf"]:
            dot.edge(str(node["id"]), str(node["left"]), label="True")
            dot.edge(str(node["id"]), str(node["right"]), label="False")
    return dot

def predict_from_tree(tree, input_features):
    """
    Walks the tree for a single input and returns (probabilities, path).
    If any node or input is invalid, logs an error to Streamlit and returns ({}, path_so_far).
    """
    # Build a dict for quick node lookup
    nodes = {node["id"]: node for node in tree["nodes"]}

    # 1) Find the root (depth == 0, not a leaf)
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

    # 2) Traverse until you hit a leaf
    while not current.get("is_leaf", False):
        # 2a) Validate node has 'feature' & 'threshold'
        if "feature" not in current or "threshold" not in current:
            st.error(f"❌ Node {current.get('id')} is missing 'feature' or 'threshold'.")
            return {}, path

        # 2b) Parse them safely
        try:
            feature_idx = int(current["feature"])
            threshold = float(current["threshold"])
        except (TypeError, ValueError):
            st.error(f"❌ Invalid 'feature' or 'threshold' at node {current['id']}.")
            return {}, path

        # 2c) Bounds‐check the feature index
        if feature_idx < 0 or feature_idx >= len(input_features):
            st.error(
                f"❌ Feature index {feature_idx} out of range "
                f"(you have {len(input_features)} features)."
            )
            return {}, path

        # 2d) Cast your input value to float
        raw_val = input_features[feature_idx]
        try:
            feature_value = float(raw_val)
        except (TypeError, ValueError):
            st.error(f"❌ Invalid feature value for index {feature_idx}: {raw_val}")
            return {}, path

        # 2e) Decide branch
        next_id = current["left"] if feature_value <= threshold else current["right"]
        path.append(next_id)

        # 2f) Lookup the next node
        current = nodes.get(next_id)
        if current is None:
            st.error(f"❌ Could not find node with id {next_id} in tree JSON.")
            return {}, path

    # 3) We’re at a leaf → return its probabilities
    return current.get("probabilities", {}), path


def visualize_tree_with_path(tree_data, feature_names, path):
    dot = graphviz.Digraph()
    for node in tree_data["nodes"]:
        if node.get("is_leaf", False):
            probs = node.get("probabilities", {})
            prob_str = "\n".join([f"Class {cls}: {prob*100:.1f}%" for cls, prob in probs.items()])
            label = f"Leaf {node['id']}\n{prob_str}"
            if node["id"] in path:
                dot.node(str(node["id"]), label, shape='box', style='filled', color='red')
            else:
                dot.node(str(node["id"]), label, shape='box', style='filled', color='lightgreen')
        else:
            feature_idx = node["feature"]
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
            label = f"{feature_name} ≤ {node['threshold']:.2f}"
            if node["id"] in path:
                dot.node(str(node["id"]), label, shape='ellipse', style='filled', color='red')
            else:
                dot.node(str(node["id"]), label, shape='ellipse', style='filled', color='lightblue')
    for node in tree_data["nodes"]:
        if not node.get("is_leaf", False):
            left_id = node["left"]
            right_id = node["right"]
            if node["id"] in path and left_id in path:
                dot.edge(str(node["id"]), str(left_id), label="True", color="red", penwidth="2")
            else:
                dot.edge(str(node["id"]), str(left_id), label="True")
            if node["id"] in path and right_id in path:
                dot.edge(str(node["id"]), str(right_id), label="False", color="red", penwidth="2")
            else:
                dot.edge(str(node["id"]), str(right_id), label="False")
    return dot


# ----------------------
# Helper Function to Rebuild Tree Mapping
# ----------------------
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






def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame where many columns are object dtype,
    strip whitespace, coerce common missing markers to NaN,
    then infer & convert each column to int or float where possible.
    """
    # 1) Strip whitespace from strings
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # 2) Replace blank strings or '?' with NaN
    df = df.replace({'': np.nan, '?': np.nan})

    # 3) For each object column, try to convert to numeric
    for col in df.columns:
        if df[col].dtype == object:
            # Attempt numeric conversion (floats or ints)
            converted = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].notna()  # where original had something

            # If every non-null original is now numeric
            if converted[mask].notna().all():
                # Check if all (non-na) values are whole numbers
                non_na = converted.dropna()
                if (non_na % 1 == 0).all():
                    # Use pandas’ nullable integer dtype
                    df[col] = converted.astype("Int64")
                else:
                    df[col] = converted
            # else: leave df[col] as object

    return df

# ----------------------
# Build mapping for tree selection (for visualization tabs)
# ----------------------
# tree_files = sorted([f for f in os.listdir() if f.startswith("tree_") and f.endswith(".json")])
# label_to_tree_id = {}
# for filename in tree_files:
#     try:
#         tree_id = int(filename.split("_")[1].split(".")[0])
#     except ValueError:
#         continue
#     data = load_tree(tree_id)
#     stats = data["stats"]
#     label = (f"Tree {tree_id} | Nodes: {stats['num_nodes']} | Leaves: {stats['num_leaves']} | "
#               f"Depth: {stats['max_depth']} | Accuracy: {stats['accuracy']:.2%}")
#     label_to_tree_id[label] = tree_id

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
    st.header("Train SMC Trees Model")
    
    # File uploader for CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="train_csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.replace("?", np.nan, inplace=True)
        df = infer_and_convert_types(df)
        st.write("Dataset Preview (first 10 rows):")
        st.dataframe(df.head(10))
        
        missing_summary = df.isna().sum()
        st.write("Missing Values Summary:")
        st.write(missing_summary)
        
        if missing_summary.sum() > 0:
            with st.expander("Missing Value Handling Options"):
                cleaning_option = st.selectbox(
                    "Handling Method", 
                    ["Do Nothing", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Specific Value"]
                )
                if cleaning_option == "Drop Rows":
                    df_clean = df.dropna()
                elif cleaning_option == "Fill with Mean":
                    df_clean = df.copy()
                    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif cleaning_option == "Fill with Median":
                    df_clean = df.copy()
                    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif cleaning_option == "Fill with Specific Value":
                    specific_val = st.number_input("Enter the specific value to fill missing values", value=0.0)
                    df_clean = df.fillna(specific_val)
                else:
                    df_clean = df
                st.write("Cleaned Dataset Preview (first 10 rows):")
                st.dataframe(df_clean.head())
        else:
            df_clean = df
            # After df_clean is defined:

        
        all_columns = list(df_clean.columns)
        target_column = st.selectbox("Select Target Column", all_columns)
        feature_columns = [col for col in all_columns if col != target_column]
        st.write("Features used for training:", feature_columns)
        
        # Save only the predictor (feature) names for later use
        with open("feature_names.json", "w") as f:
            json.dump(feature_columns, f, indent=2)
        
        st.subheader("SMC Training Parameters")
        tree_size = st.number_input("Tree Size (a)", min_value=1, value=10, step=1, help="Size of the tree.")
        num_iterations = st.number_input("Number of Iterations", min_value=1, value=10, step=1, help="Positive integer for iterations.")
        num_trees = st.number_input("Number of Trees", min_value=1, value=5, step=1, help="Positive integer for number of trees.")
        resampling_options = ["residual", "systematic", "knapsack", "min_error", "variational", "min_error_imp", "CIR"]
        resampling_scheme = st.selectbox("Resampling Scheme", resampling_options)
        
        # Save the cleaned dataset to a temporary path for training
        csv_path = f"datasets/{uploaded_file.name}"
        os.makedirs("datasets", exist_ok=True)
        df_clean.to_csv(csv_path, index=False)
        
        if st.button("Train SMC Model"):
            # Clear previous tree JSON files in the MODELS_DIR
            for filename in os.listdir(MODELS_DIR):
                if filename.startswith("tree_") and filename.endswith(".json"):
                    os.remove(os.path.join(MODELS_DIR, filename))
        
            with st.spinner("Training in progress..."):
                accuracy = train_smc_model(csv_path, target_column, tree_size, num_iterations, num_trees, resampling_scheme)
            if accuracy is not None:
                st.success(f"SMC Training complete. Ensemble Accuracy: {accuracy:.2%}")
                X_train, X_test, y_train, y_test = train_test_split(df_clean[feature_columns], df_clean[target_column], test_size=0.30, random_state=42)
                st.session_state["X_test"] = X_test.to_numpy()
                st.session_state["y_test"] = y_test.to_numpy()
                st.cache_data.clear()
                st.write("SMC trees saved in the 'models' folder and test data stored for evaluation.")
            else:
                st.error("SMC training failed.")




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
    # Rebuild mapping using the helper function:
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
# Tab 3: Feature Importance
# ----------------------
with tab3:
    st.header("Feature Importance across all SMC Trees")
    with open("feature_importance.json", "r") as f:
        importance_dict = json.load(f)
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)
    fig, ax = plt.subplots(figsize=(8, max(4, len(features)*0.4)))
    ax.barh(features[::-1], [imp*100 for imp in importances[::-1]], color="skyblue")
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title("Feature Importance based on Frequency of Usage")
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Importance Values")
    st.dataframe({"Feature": features, "Importance (%)": [round(imp*100,2) for imp in importances]}, use_container_width=True)  

# ----------------------
# Tab 4: Interactive Prediction (Exclude "Target")
# ----------------------
# ----------------------
# Tab 4: Interactive Prediction (Simplified & Improved Voting)
# ----------------------
with tab4:
    st.header("🎯 Interactive Prediction")

    all_feature_names = load_feature_names()
    feature_names_for_prediction = [name for name in all_feature_names if name.lower() != "target"]

    st.subheader("📝 Enter Input Values for Features")

    # Use a compact input table
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
                predicted_class = max(pred_probs, key=pred_probs.get)
                confidence = pred_probs[predicted_class]

                st.success(f"**Predicted Class:** `{predicted_class}`  \n**Confidence:** `{confidence:.2%}`")

                st.subheader("Class Probabilities")
                st.dataframe(pd.DataFrame.from_dict(pred_probs, orient="index", columns=["Probability (%)"]).applymap(lambda x: round(x * 100, 2)))

                st.subheader("Tree Path Highlight")
                st.graphviz_chart(visualize_tree_with_path(tree_data, all_feature_names, path))

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
                    top_class = max(pred_probs, key=pred_probs.get)
                    vote_counter[top_class] = vote_counter.get(top_class, 0) + 1

                total_votes = sum(vote_counter.values())
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
from streamlit_plotly_events import plotly_events

with tab5:
    st.header("Overall Performance Analysis")

    tree_files = get_valid_tree_files()
    if not tree_files:
        st.warning("No tree JSON files found. Please train your model first.")
    else:
        # Load stats
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

        # Plot
        fig = px.scatter(
            df_stats,
            x="Nodes",
            y="Accuracy",
            hover_name="Tree ID",
            title="Tree Accuracy vs Tree Size",
            labels={"Nodes": "Tree Size (Number of Nodes)", "Accuracy": "Accuracy"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Selectbox to show a tree
        selected_id = st.selectbox(
            "Select a Tree to Visualize",
            df_stats["Tree ID"].astype(str)
        )
        selected_tree = load_tree(int(selected_id))
        feature_names = load_feature_names()
        st.graphviz_chart(visualize_tree(selected_tree, feature_names))

        # Ensemble accuracy
        if "X_test" in st.session_state and "y_test" in st.session_state:
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]

            ensemble_predictions = []
            for x in X_test:
                votes = []
                for tree_id in df_stats["Tree ID"]:
                    tree_data = load_tree(tree_id)
                    pred_probs, _ = predict_from_tree(tree_data, list(x))
                    pred_label = max(pred_probs.items(), key=lambda p: p[1])[0]
                    votes.append(pred_label)
                majority = max(set(votes), key=votes.count)
                ensemble_predictions.append(majority)

            ensemble_accuracy = np.mean([str(p) == str(t) for p, t in zip(ensemble_predictions, y_test)])
        else:
            ensemble_accuracy = None

        # Summary
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



# with tab5:
#     st.header("Overall Performance Analysis")

#     tree_files = get_valid_tree_files()
#     if not tree_files:
#         st.warning("No tree JSON files found. Please train your model first.")
#     else:
#         # Extract statistics
#         accuracies, depths, num_nodes, num_leaves, tree_labels, tree_ids = [], [], [], [], [], []
#         for filename in tree_files:
#             with open(os.path.join(MODELS_DIR, filename), "r") as file:
#                 data = json.load(file)
#                 stats = data.get("stats", {})
#                 tree_id = int(filename.replace("tree_", "").replace(".json", ""))
#                 tree_ids.append(tree_id)
#                 tree_labels.append(f"Tree {tree_id}")
#                 accuracies.append(stats.get("accuracy", np.nan))
#                 depths.append(stats.get("max_depth", np.nan))
#                 num_nodes.append(stats.get("num_nodes", np.nan))
#                 num_leaves.append(stats.get("num_leaves", np.nan))

#         # DataFrame for plotting
#         performance_df = pd.DataFrame({
#             "Tree": tree_labels,
#             "Accuracy": accuracies,
#             "Depth": depths,
#             "Nodes": num_nodes,
#             "Leaves": num_leaves,
#             "Tree_ID": tree_ids
#         })

#         # Interactive scatter plot
#         fig = px.scatter(
#             performance_df,
#             x="Nodes",
#             y="Accuracy",
#             hover_data=["Tree", "Depth", "Leaves"],
#             labels={"Nodes": "Tree Size (Number of Nodes)", "Accuracy": "Accuracy"},
#             title="Tree Accuracy vs. Tree Size",
#             custom_data=["Tree_ID"]
#         )

#         fig.update_layout(
#             xaxis_title="Tree Size (Number of Nodes)",
#             yaxis_title="Accuracy",
#             hovermode="closest"
#         )

#         # Capture click event
#         selected_point = st.plotly_chart(fig, use_container_width=True, click_event=True)

#         if selected_point:
#             clicked_points = selected_point.get("points", [])
#             if clicked_points:
#                 clicked_tree_id = clicked_points[0]["customdata"][0]
#                 st.subheader(f"Visualization of Tree {clicked_tree_id}")
#                 tree_data = load_tree(clicked_tree_id)
#                 if tree_data:
#                     feature_names = load_feature_names()
#                     st.graphviz_chart(visualize_tree(tree_data, feature_names))
#                 else:
#                     st.error(f"Tree {clicked_tree_id} data could not be loaded.")
#         else:
#             st.info("Click a dot in the plot above to visualize the corresponding tree.")

#         # Compute ensemble majority-vote accuracy
#         if "X_test" in st.session_state and "y_test" in st.session_state:
#             X_test = st.session_state["X_test"]
#             y_test = st.session_state["y_test"]

#             ensemble_predictions = []
#             for x in X_test:
#                 preds = []
#                 for tree_id in tree_ids:
#                     tree_data = load_tree(tree_id)
#                     pred_probs, _ = predict_from_tree(tree_data, list(x))
#                     # Get prediction class with highest probability
#                     pred_class = max(pred_probs.items(), key=lambda item: item[1])[0]
#                     preds.append(pred_class)
#                 # Majority vote
#                 final_pred = max(set(preds), key=preds.count)
#                 ensemble_predictions.append(final_pred)

#             ensemble_accuracy = np.mean([pred == str(y_true) for pred, y_true in zip(ensemble_predictions, y_test)])

#             st.subheader("Summary Statistics")
#             st.markdown(f"""
#             - **Ensemble (Majority Voting) Accuracy:** {ensemble_accuracy:.2%}
#             - **Average Max Depth:** {performance_df['Depth'].mean():.2f}
#             - **Average Number of Nodes:** {performance_df['Nodes'].mean():.2f}
#             - **Average Number of Leaves:** {performance_df['Leaves'].mean():.2f}
#             """)
#         else:
#             st.warning("Test data not found. Retrain the model (Tab 0) to generate test data for ensemble accuracy.")



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
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf

with  tab7:
   
    st.header("Statistical Tests")
    st.write("Upload a CSV file for running statistical tests on your data.")
    
    # Upload CSV file
    uploaded_stat_file = st.file_uploader("Choose a CSV file", type=["csv"], key="stat_tests_file")
    
    if uploaded_stat_file is not None:
        df_stat = pd.read_csv(uploaded_stat_file)
        df_stat = df_stat.dropna()
        st.write("Dataset Preview:")
        st.dataframe(df_stat.head(10))
        
        # Choose the statistical test
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
# Tab 7: Custom Plotting
# ----------------------
import pandas as pd
import matplotlib.pyplot as plt

with tab8:
    st.header("Custom Plotting")
    st.write("Upload a CSV file or use an existing one to plot your features.")

    # File uploader for plotting dataset.
    uploaded_plot_file = st.file_uploader("Choose a CSV file for plotting", type=["csv"], key="plot_file")
    
    if uploaded_plot_file is not None:
        df_plot = pd.read_csv(uploaded_plot_file)
        df_plot = df_plot.dropna()  # Optional: remove missing values
        st.dataframe(df_plot.head(10))
        
        # Choose a plot type
        plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram"], key="plot_type")
        
        # Based on the plot type, choose which columns to plot
        if plot_type in ["Scatter Plot", "Line Plot", "Bar Plot"]:
            x_col = st.selectbox("Select X-axis Feature", df_plot.columns, key="plot_x")
            y_col = st.selectbox("Select Y-axis Feature", df_plot.columns, key="plot_y")
        elif plot_type == "Histogram":
            col = st.selectbox("Select Feature for Histogram", df_plot.columns, key="plot_hist")
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
        
        # Create the plot
        fig, ax = plt.subplots()
        if plot_type == "Scatter Plot":
            ax.scatter(df_plot[x_col], df_plot[y_col], color="dodgerblue")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        elif plot_type == "Line Plot":
            ax.plot(df_plot[x_col], df_plot[y_col], marker="o", linestyle="-", color="darkorange")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Line Plot: {x_col} vs {y_col}")
        elif plot_type == "Bar Plot":
            ax.bar(df_plot[x_col], df_plot[y_col], color="mediumseagreen")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Bar Plot: {x_col} vs {y_col}")
        elif plot_type == "Histogram":
            ax.hist(df_plot[col], bins=bins, edgecolor="black", color="orchid")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram of {col}")
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please upload a CSV file to create plots.")

# --- Evaluation Metrics Tab ---
with tab9:
    st.header("Evaluation Metrics (ROC & AUC)")

    # Check if test data is available in session state
    if "X_test" not in st.session_state or "y_test" not in st.session_state:
        st.error("Test data not found. Please train the model first (Tab 0).")
    else:
        # Retrieve test data from session state
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        # Let user choose prediction mode for evaluation
        eval_mode = st.radio("Prediction Mode for Evaluation", options=["Single Tree", "Ensemble"], key="eval_mode")

        # List to store predicted probability for the positive class (assumed label "1")
        y_scores = []

        # Use the same label_to_tree_id and helper functions from your app.
        # Assuming these functions are defined above in your app:
        # load_tree(tree_id) and predict_from_tree(tree, input_features)
        if eval_mode == "Single Tree":
            selected_label_eval = st.selectbox("Select Tree for Evaluation", list(label_to_tree_id.keys()), key="eval_tree")
            tree_id_eval = label_to_tree_id[selected_label_eval]
            tree_data_eval = load_tree(tree_id_eval)
            for row in X_test:
                # Predict returns (prediction, path); we ignore the path here.
                pred, _ = predict_from_tree(tree_data_eval, list(row))
                # Assume the positive class is labeled "1"
                y_scores.append(pred.get("1", 0))
        else:  # Ensemble mode
            all_tree_files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith("tree_") and f.endswith(".json")])
            for row in X_test:
                ensemble_preds = []
                for file in all_tree_files:
                    try:
                        tid = int(file.split("_")[1].split(".")[0])
                    except ValueError:
                        continue
                    tree_data = load_tree(tid)
                    pred, _ = predict_from_tree(tree_data, list(row))
                    ensemble_preds.append(pred.get("1", 0))
                if ensemble_preds:
                    ensemble_avg = sum(ensemble_preds) / len(ensemble_preds)
                else:
                    ensemble_avg = 0
                y_scores.append(ensemble_avg)
        
        # Compute ROC curve and AUC
        # Make sure y_test is numeric (e.g., contains 0 and 1)
        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        st.write(f"ROC AUC: {roc_auc:.3f}")
        
        # Plot ROC curve
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        st.pyplot(fig)