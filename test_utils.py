import pandas as pd
import numpy as np


def dict_to_df(data_dict):
    processed_data = []
    for inventor_id, patents_dict in data_dict.items():
        for patent_id, data in patents_dict.items():
            processed_data.append({
                'inventor_id': inventor_id,
                'patent_id': patent_id,
                'encoded_feature': data
            })

    # Create a DataFrame from the processed data
    df = pd.DataFrame(processed_data)

    return df


def dict_to_dataframe(data_dict):
    processed_data = []
    for inventor_id, patents_dict in data_dict.items():
        for patent_id, data in patents_dict.items():
            processed_data.append({
                'inventor_id': inventor_id,
                'patent_id': patent_id,
                'title': data['title'],
                'abstract': data['abstract'],
                'grant_date': data['date'],
                'first_name': data['first_name'],
                'last_name': data['last_name'],
                'assignee': data['assignee'],
                'assignee_longitude': data['assignee_longitude'],
                'assignee_latitude': data['assignee_latitude'],
                'assignee_country': data['assignee_country'],
                'co_inventors': ', '.join(data['co_inventors']),
                'wipos': data['wipos']
            })

    # Create a DataFrame from the processed data
    df = pd.DataFrame(processed_data)

    return df


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def visualize_embeddings(embeddings_combined, inventor_ids):
    # Calculate the Silhouette Coefficient
    silhouette_avg = silhouette_score(embeddings_combined, inventor_ids)
    print("Average Silhouette Coefficient:", silhouette_avg)

    # Perform dimensionality reduction (e.g., t-SNE)
    tsne = TSNE(n_components=2, perplexity=30, random_state=88)
    embeddings_tsne = tsne.fit_transform(embeddings_combined)

    # Create a mapping dictionary for inventor IDs to numeric labels
    unique_ids = np.unique(inventor_ids)
    label_map = {id: label for label, id in enumerate(unique_ids)}

    # Convert the inventor_ids to numeric labels using the mapping dictionary
    inventor_labels = np.array([label_map[id] for id in inventor_ids])

    # Generate a color map for different inventor IDs
    num_ids = len(unique_ids)
    color_map = plt.cm.get_cmap('tab10', num_ids)

    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Plot the embeddings with different colors based on inventor ID labels
    sc = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=inventor_labels, cmap=color_map)

    # Create a legend with the inventor IDs
    legend_handles = []
    for id in unique_ids:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(label_map[id]), markersize=5))
    plt.legend(legend_handles, unique_ids, loc='upper left')

    plt.title('Embeddings Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the figure as a PNG image
    plt.savefig('results_plot.png')

    # Show the plot
    plt.show()


