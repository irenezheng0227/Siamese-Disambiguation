import pandas as pd
import numpy as np
import random
from itertools import combinations, product

def generate_positive_dataset(dictionary):
    # Initialize lists to store the anchor, positive, and negative samples
    anchors = []
    positives = []
    negatives = []
    labels = []

    # Iterate over each inventor
    for inventor_id, patents in dictionary.items():
        # Generate all combinations of patents within the same inventor_id group
        patent_combinations = combinations(patents.keys(), 2)

        # Add anchor-positive pairs
        for pid_1, pid_2 in patent_combinations:
            anchors.append({pid_1: patents[pid_1]})
            positives.append({pid_2: patents[pid_2]})
            labels.append(1) # 1 for the positive pairs
    
    # Create a DataFrame from the samples
    test_df = pd.DataFrame({'Patent1': anchors, 'Patent2': positives, 'Label': labels})

    # Return the triplet DataFrame
    return test_df

def generate_negative_dataset(dictionary):
    # Initialize lists to store the anchor and negative samples
    anchors = []
    negatives = []
    labels = []

    # Generate all combinations of patents with different inventor_ids
    inventor_combinations = product(dictionary.keys(), repeat=2)

    # Iterate over each combination of inventor_ids
    for iid_1, iid_2 in inventor_combinations:
        # Skip combinations where the inventor_ids are the same
        if iid_1 == iid_2:
            continue

        # Iterate over each patent combination within the inventor_id pairs
        for pid_1, pid_2 in product(dictionary[iid_1].keys(), dictionary[iid_2].keys()):
            #labels.append(1) # 1 for the positive pairs
            anchors.append({pid_1: dictionary[iid_1][pid_1]})
            negatives.append({pid_2: dictionary[iid_2][pid_2]})
            labels.append(0)  # 0 for the pairs with different inventor_ids

    # Create a DataFrame from the samples
    test_df = pd.DataFrame({'Patent1': anchors, 'Patent2': negatives, 'Label': labels})

    # Return the triplet DataFrame
    return test_df

def generate_triplet_dataset(dictionary):
    # Initialize lists to store the anchor, positive, and negative samples
    anchors = []
    positives = []
    negatives = []

    # Iterate over each inventor
    for inventor_id, patents in dictionary.items():
        # Generate all combinations of patents within the same inventor_id group
        patent_combinations = combinations(patents.keys(), 2)

        # Add anchor-positive pairs
        for pid_1, pid_2 in patent_combinations:
            anchors.append({pid_1: patents[pid_1]})
            positives.append({pid_2: patents[pid_2]})

            # Select a random inventor_id other than the current one
            random_iid = random.choice([iid for iid in dictionary.keys() if iid != inventor_id])

            # Select a random patent from the randomly selected inventor_id group as the negative sample
            pid_3 = random.choice(list(dictionary[random_iid].keys()))
            negatives.append({pid_3: dictionary[random_iid][pid_3]})

    # Create a DataFrame from the samples
    triplet_df = pd.DataFrame({'Anchor': anchors, 'Positive': positives, 'Negative': negatives})

    # Return the triplet DataFrame
    return triplet_df

def create_inventor_dictionary(df):
    # Group the DataFrame by 'person_id' and create a dictionary of grouped data
    grouped_data = df.groupby('person_id')

    # Initialize an empty dictionary to store the result
    inventor_dict = {}

    # Iterate over each group
    for inventor_id, group in grouped_data:
        # Create a nested dictionary for each inventor
        inventor_dict[inventor_id] = {}

        # Iterate over each row in the group
        for index, row in group.iterrows():
            # Extract the appln_id and encoded_feature from the row
            appln_id = row['appln_id']
            encoded_feature = row['encoded_feature']

            # Add the appln_id and encoded_feature to the nested dictionary
            inventor_dict[inventor_id][appln_id] = encoded_feature

    return inventor_dict

# convert word vector(a column from a tsv file) to a cleaned numpy array
def preprocess_data(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    
    # Make a copy of the dataframe to avoid changing the original one
    df_copy = df.copy()

    for i in range(len(df_copy[column_name])):

        data = df_copy[column_name][i].strip('[]').replace('][', ' ')

        # Split by spaces
        data = data.split()

        # Convert strings to float32
        data = np.array(data, dtype=np.float32)

        # Assign the preprocessed data back to the DataFrame
        df_copy.at[i, column_name] = data

    return df_copy
