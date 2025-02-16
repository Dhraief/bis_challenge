import pandas as pd
import pickle
import torch
import os
from collections import Counter
import logging
import json

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#GLOABL VARIABLES
N_ADDED_NODES= 3761
CONFIG_PATH =  "../../config/config.json"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        config = json.load(f)

    input_base = config["base_paths"]["input"]
    output_base = config["base_paths"]["output"]

    # Dynamically construct full paths
    paths = {}
    for key, file_info in config["files"].items():
        base_path = input_base if file_info["type"] == "input" else output_base
        paths[key] = os.path.join(base_path, file_info["name"])

    # Include base paths
    paths["input_path"] = input_base
    paths["output_path"] = output_base

    return paths

config = load_config()
NODE_FEATURES_PATH = config["node_features"]
MAPPINGS_NODE_PATH = config["mappings_node"]
MAPPING_EDGE_PATH = config["mapping_edge"]
EDGES_PATH = config["edges"]
EDGE_FEATURES_PATH = config["edge_features"]
ACCOUNT_DATASET_PATH = config["account_dataset"]
TRAIN_TARGET_PATH = config["train_target"]
TRANSACTION_DATASET_PATH = config["transaction_dataset"]
OUTPUT_PATH = config["output_path"]



def create_column_mapping(df, column_name):
    """
    Creates a mapping for a categorical column and returns a dictionary.
    """
    unique_values = df[column_name].dropna().unique()
    return {value: idx for idx, value in enumerate(unique_values, start=1)}

def unique_values(lst):
    return [key for key, count in Counter(lst).items() if count == 1]

def get_node_features(data,n):
    def generate_node_features(data,n):
        logger.info("Generating node features...")
        data["account_id"] = data["account_id"].astype(int)
        data["age"] = data["age"].apply(lambda x: round(float(x)))
        data["initial_balance"] = data["initial_balance"].apply(lambda x: round(float(x)))
        
        mappings = {col: create_column_mapping(data, col) for col in ["assigned_bank_type", "assigned_bank"]}
        
        for col, mapping in mappings.items():
            data[f"{col}_mapped"] = data[col].map(mapping).astype(int)
        
        data.sort_values(by=["account_id"], inplace=True)
        num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        node_features = torch.tensor(data[num_cols].values)
        # append n zero rows to node_features
        zero_rows = torch.zeros((n, node_features.shape[1]), dtype=torch.int64)
        node_features = torch.cat((node_features, zero_rows), dim=0)
        logger.info("Node features generated.")

        with open(NODE_FEATURES_PATH, "wb") as f:
            pickle.dump(node_features, f)
        with open(MAPPINGS_NODE_PATH, "wb") as f:
            pickle.dump(mappings, f)
        
        return node_features
    

    if not os.path.exists(NODE_FEATURES_PATH):
        return generate_node_features(data,n),
    with open(NODE_FEATURES_PATH, "rb") as f:
        return pickle.load(f)
    

def get_edge_feature(train_target, transaction,n_edges):
    def generate_edge_feature(train_target, transaction,n_edges):
        def generate_illicit_features(train_target, transaction,edge_feature):
            l = transaction.transaction_id.unique().tolist()
            l = sorted(l)
            order_map = {v:i for i, v in enumerate(l)}
            train_target["transaction_id_mapped"] = train_target["transaction_id"].map(order_map)
            illicit_transactions = train_target["transaction_id_mapped"].tolist()

            #schema_type = train_target["launderi ng_schema_type"].apply(lambda x: int(x.split("_")[-1])).values
                    # Extract laundering schema type and convert to PyTorch tensor
            schema_type = torch.tensor(
                train_target["laundering_schema_type"].apply(lambda x: int(x.split("_")[-1])).values,
                dtype=torch.int  # Ensure it matches the tensor type of edge_feature
            )

            edge_feature[illicit_transactions, 0] = 1
            edge_feature[illicit_transactions, 1] = schema_type
            return edge_feature
        
        def generate_all_features(train_target, transaction,edge_feature):
            path_df_tmp = OUTPUT_PATH+"df_temp.pkl"
            if os.path.exists(path_df_tmp):
                # load pickle using pickle.load
                with open(path_df_tmp, "rb") as f:
                    df = pickle.load(f)

            else:
                df =  transaction.drop_duplicates(subset=['transaction_id'], keep='first')
                weekday_mapping = {
                    'Monday': 0,
                    'Tuesday': 1,
                    'Wednesday': 2,
                    'Thursday': 3,
                    'Friday': 4,
                    'Saturday': 5,
                    'Sunday': 6
                }

                df['weekday_mapped'] = df['weekday'].map(weekday_mapping)
                df.sort_values(by=['transaction_id'], inplace=True)
                cols_to_map = ['channel', 'payment_system', 'category_0', 'category_1', 'category_2' ]
                mappings = {col: create_column_mapping(df, col) for col in cols_to_map}
                for col, mapping in mappings.items():
                    df[f"{col}_mapped"] = df[col].map(mapping).astype(int)
                
                df['amount'] = df['amount'].apply(lambda x: round(float(x)))

                
                cols_num = ["month","day","weekday_mapped","hour","min"	,"sec",	"channel_mapped","payment_system_mapped","category_0_mapped","category_1_mapped","category_2_mapped","amount"]
#                df[cols_num].to_pickle(path_df_tmp)
                with open(path_df_tmp, "wb") as f:
                    pickle.dump(df, f)

            val_np = df[cols_num].values      
            edge_feature = torch.tensor(df[cols_num].values, dtype=torch.int32)
            
            return edge_feature
        
        edge_feature = torch.zeros((n_edges, 3), dtype=torch.int32)
        logger.info("Edges features illicit generating...")

        if os.path.exists("edges_features_temp.pkl"):
            with open("edges_features_temp.pkl", "rb") as f:
                edge_feature = pickle.load(f)
        else:
            edge_feature = generate_illicit_features(train_target, transaction,edge_feature)
            with open("edges_features_temp.pkl", "wb") as f:
                pickle.dump(edge_feature, f)

        # edge_feature = generate_illicit_features(train_target, transaction,edge_feature)
        logger.info("Edges features illicit generated.")

        logger.info("Edges features all generating...")
        edge_feature_all = generate_all_features(train_target, transaction,edge_feature)
        logger.info("Edges features all generated.")

        logger.info("Edges features concatenating...")
        edge_feature = torch.cat((edge_feature, edge_feature_all), dim=1)
        logger.info("Edges features illicit generated.")

        with open(EDGES_PATH, "wb") as f:
            pickle.dump(edge_feature, f)
        
        logger.info("Edges saved.")
        return edge_feature



    if not os.path.exists(EDGE_FEATURES_PATH):
        res =  generate_edge_feature(train_target, transaction,n_edges)
        with open(EDGE_FEATURES_PATH, "wb") as f:
            pickle.dump(res, f)

    with open(EDGE_FEATURES_PATH, "rb") as f:
        return pickle.load(f)

def get_edges(df):
    def generate_edges(df):
        def create_mapping(df, column_name):
                counterparlist = list(df[column_name].unique())
                num_counterpart = []
                non_num_counterpart = []
                for l in counterparlist:
                    if l.isnumeric():
                        num_counterpart.append(l)
                    else:
                        non_num_counterpart.append(l)

                mapping = {}
                for num in num_counterpart:
                    mapping[num] = int(num)

                for non_num in non_num_counterpart:
                    mapping[non_num] = len(mapping) + 1
                return mapping, len(non_num_counterpart)

            
        edge_df = df[["transaction_id", "account_id", "counterpart_id", "transaction_direction"]]
        
        logger.info("Generating edge mappings...")
        mapping, n_added_nodes = create_mapping(df, "counterpart_id")
        logger.info("Mapping generated.")
        

        edge_df["counterpart_id_map"] = edge_df["counterpart_id"].astype(str).map(mapping).astype(int)
        edge_df["account_id"] = edge_df["account_id"].astype(int)
        
        #logger.info("Generating unique transactions...")
        #unique_transactions = unique_values(edge_df["transaction_id"].tolist())
        #logger.info("Unique transactions generated.")
        
        logger.info("Getting inbound and outbound edges...")
        edge_df_outbound = edge_df[edge_df["transaction_direction"] == "outbound"]
        outbound_transactions = edge_df_outbound["transaction_id"].unique().tolist()
        edge_df_inbound = edge_df[edge_df["transaction_direction"] == "inbound"]
        edge_df_inbound = edge_df_inbound[~edge_df_inbound["transaction_id"].isin(outbound_transactions)]

        
        cols_to_keep = ["transaction_id", "account_id", "counterpart_id_map"]
        logger.info("Concatenating inbound and outbound edges...")
        edge_df_concat = pd.concat([edge_df_outbound[cols_to_keep], edge_df_inbound[cols_to_keep]], axis=0)

        logger.info("Sorting edges by transaction_id...")
        edge_df_concat.sort_values(by=["transaction_id"], inplace=True)
        logger.info("Edges sorted.")
        
        with open("edge_df_concat.pkl", "wb") as f:
            pickle.dump(edge_df_concat, f)
        
        logger.info("Edges generated.")
        edge_index = torch.tensor(edge_df_concat[["account_id", "counterpart_id_map"]].values)
        with open(EDGES_PATH, "wb") as f:
            pickle.dump(edge_index, f)
            
        with open(MAPPING_EDGE_PATH, "wb") as f:
            pickle.dump(mapping, f)
        logger.info("Edges saved.")

        n_edges = df.transaction_id.nunique()
        assert edge_index.shape[0] == n_edges, "Edge index must have the same number of rows as the number of unique transactions."
        return edge_index,n_added_nodes

    if not os.path.exists(EDGES_PATH):
        return generate_edges(df)
    with open(EDGES_PATH, "rb") as f:
        return pickle.load(f),N_ADDED_NODES

def main():
    logger.info("Reading datasets...")
    account_dataset = pd.read_parquet(ACCOUNT_DATASET_PATH)
    train_target = pd.read_parquet(TRAIN_TARGET_PATH)
    transaction = pd.read_parquet(TRANSACTION_DATASET_PATH)
    # logger.info("Data loaded.")

    logger.info("Generating edges...")
    edge_index,n_added_nodes = get_edges(transaction)
    print(n_added_nodes)
    logger.info("Edges generated.")
    
    logger.info("Generating node features...")
    n_added_nodes = N_ADDED_NODES
    node_features = get_node_features(account_dataset,n_added_nodes)
    logger.info("Node features generated.")

    logger.info("Generating edges features...")
    edge_features = get_edge_feature(train_target, transaction, edge_index.shape[0])
    logger.info("Graph data processing complete.")
    num_nodes = node_features.shape[0]
    num_edges = edge_index.shape[1]

    assert num_edges > 0, "The graph should have at least one edge"
    assert num_nodes > 0, "The graph should have at least one node"
    assert edge_index.max() < num_nodes, "Edge index contains node IDs that exceed the number of nodes"
    assert edge_index.shape[0]==edge_features.shape[0]

    # assert edge_index.shape[0] == edge_features.shape[0], "Edge index and edge features must have the same number of rows."

if __name__ == "__main__":
    main()
