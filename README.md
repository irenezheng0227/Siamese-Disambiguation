# Siamese-Disambiguation
Siamese Networks for Name Disambiguation

# Kaggle link: 
https://www.kaggle.com/code/irenezheng227/siamese-networks-for-name-disambiguation/edit

Feature: patent_abstract_encoded

4 base models: MLP, CNN, LSTM, RNN

2 different similarity calculations: Euclidean Distance, Cosine Similarity

Metrics: Silhouette Score, PCA Graph (Clustering)
         Accuracy, F1 score, AUC, Precision (Pairwise comparison)

Version 2 employs a different data preparation process, where each patent serves as an index in the anchor dataset and 
is associated with its respective positive and negative indices. This approach tries to maximize the utilization of the input dataset.






