import torch
import torch.distributed as dist


def pairwise_similarity(outputs, temperature=0.5, multi_gpu=False, adv_type='None'):
    '''
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity

    '''
    B = outputs.shape[0]
    outputs_norm = outputs / (outputs.norm(dim=1).view(B, 1) + 1e-8)
    similarity_matrix = (1. / temperature) * torch.mm(outputs_norm, outputs_norm.transpose(0, 1).detach())
    return similarity_matrix, outputs


def NT_xent(similarity_matrix, adv_type):
    """
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    """

    N2 = len(similarity_matrix)
    if adv_type == 'None':
        N = int(len(similarity_matrix) / 2)
    elif adv_type == 'Rep':
        N = int(len(similarity_matrix) / 3)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2, N2)).cuda()

    NT_xent_loss = - torch.log(
        similarity_matrix_exp / (torch.sum(similarity_matrix_exp, dim=1).view(N2, 1) + 1e-8) + 1e-8)

    if adv_type == 'None':
        NT_xent_loss_total = (1. / float(N2)) * torch.sum(
            torch.diag(NT_xent_loss[0:N, N:]) + torch.diag(NT_xent_loss[N:, 0:N]))
    elif adv_type == 'Rep':
        NT_xent_loss_total = (1. / float(N2)) * torch.sum(
            torch.diag(NT_xent_loss[0:N, N:2 * N]) + torch.diag(NT_xent_loss[N:2 * N, 0:N])
            + torch.diag(NT_xent_loss[0:N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, 0:N])
            + torch.diag(NT_xent_loss[N:2 * N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, N:2 * N]))
    return NT_xent_loss_total
