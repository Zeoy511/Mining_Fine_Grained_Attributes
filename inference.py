import torch
import torch.nn.functional as F


def compute_partial_based_covariance_matrix(partial_visual_representations):
    """
    Compute the partial-based covariance matrix.

    :param partial_visual_representations: (Tensor) Partial visual representations of training samples.
    :return: (Tensor) Partial-based covariance matrix.
    """
    mean = torch.mean(partial_visual_representations, dim=0, keepdim=True)
    cov_matrix = torch.matmul((partial_visual_representations - mean).T, (partial_visual_representations - mean)) / (
                partial_visual_representations.shape[0] - 1)
    return cov_matrix


def compute_visual_similarity(query_pvr, class_cov_matrix):
    """
    Compute the visual attribute similarity score.

    :param query_pvr: (Tensor) Query partial visual representation.
    :param class_cov_matrix: (Tensor) Class covariance matrix.
    :return: (Tensor) Visual attribute similarity score.
    """
    similarity_score = torch.matmul(torch.matmul(query_pvr.T, class_cov_matrix), query_pvr)
    return similarity_score


def compute_semantic_similarity(query_semantic, class_semantic):
    """
    Compute the semantic attribute similarity.

    :param query_semantic: (Tensor) Query semantic attributes.
    :param class_semantic: (Tensor) Class semantic attributes.
    :return: (Tensor) Semantic attribute similarity score.
    """
    dot_product = torch.sum(query_semantic * class_semantic, dim=-1)
    norm_product = torch.norm(query_semantic, dim=-1) * torch.norm(class_semantic, dim=-1)
    similarity_score = dot_product / norm_product
    return similarity_score


def inference(query_samples, support_samples, beta=0.5):
    """
    Perform the inference process using the defined similarity metrics.

    :param query_samples: (dict) Query samples containing 'pvr' and 'semantic'.
    :param support_samples: (list of dict) Support samples, each containing 'pvr' and 'semantic'.
    :param beta: (float) Trade-off weight between visual and semantic similarities.
    :return: (Tensor) Probability distribution over the classes.
    """
    query_pvr = query_samples['pvr']
    query_semantic = query_samples['semantic']

    # Calculate the covariance matrices for each class
    class_cov_matrices = []
    class_semantics = []
    for sample in support_samples:
        pvr = sample['pvr']
        semantic = sample['semantic']
        class_cov_matrices.append(compute_partial_based_covariance_matrix(pvr))
        class_semantics.append(semantic)

    # Compute the similarity scores
    visual_similarities = [compute_visual_similarity(query_pvr, cov).mean() for cov in class_cov_matrices]
    semantic_similarities = [compute_semantic_similarity(query_semantic, sem).mean() for sem in class_semantics]

    # Combine the similarities
    combined_similarities = [beta * vs + (1 - beta) * ss for vs, ss in zip(visual_similarities, semantic_similarities)]

    # Compute the softmax probabilities
    probabilities = F.softmax(torch.tensor(combined_similarities), dim=0)

    return probabilities