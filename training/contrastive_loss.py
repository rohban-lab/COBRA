import torch
import torch.distributed as dist


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (    B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''
    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')
    return sim_matrix


def NT_xent(sim_matrix, temperature=0.5, chunk=4, eps=1e-8, opposite_pair=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''
    device = sim_matrix.device
    B = int(sim_matrix.size(0) // chunk)  # B = B' / chunk
    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix
    
    if chunk == 2:
        loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)
    elif chunk == 3:
        loss = torch.sum(sim_matrix[0:B, B:2 * B].diag() + sim_matrix[B:2 * B, 0:B].diag() +
                         sim_matrix[0:B, 2 * B:].diag() + sim_matrix[2 * B:, 0:B].diag() +
                         sim_matrix[B:2 * B, 2 * B:].diag() + sim_matrix[2 * B:, B:2 * B].diag()
                         ) / float(sim_matrix.size(0))
    elif chunk == 4:
        if opposite_pair:
            loss = (torch.sum(sim_matrix[0:B, B:2 * B].diag() + sim_matrix[B:2 * B, 0:B].diag() +
                         sim_matrix[0:B, 2 * B:3 * B].diag() + sim_matrix[2 * B:3 * B, 0:B].diag() +
                         sim_matrix[B:2 * B, 2 * B:3 * B].diag() + sim_matrix[2 * B:3 * B, B:2 * B].diag() +
                         sim_matrix[0:B, 3 * B:].diag() + sim_matrix[3 * B:, 0:B].diag() +
                         sim_matrix[B:2 * B, 3 * B:].diag() + sim_matrix[3 * B:, B:2 * B].diag() +
                         sim_matrix[2 * B:3 * B, 3 * B:].diag() + sim_matrix[3 * B:, 2 * B:3 * B].diag()
                         ) -
                        torch.sum(
                            sim_matrix[0:int(B/2), int(B/2):B].diag() + sim_matrix[int(B/2):B, 0:int(B/2)].diag() +
                            sim_matrix[int(3*B/2):2*B, B:int(3*B/2)].diag() + sim_matrix[B:int(3*B/2), int(3*B/2):2*B].diag()
                         )) / float(sim_matrix.size(0))
        else:
            loss = torch.sum(sim_matrix[0:B, B:2 * B].diag() + sim_matrix[B:2 * B, 0:B].diag() +
                            sim_matrix[0:B, 2 * B:3 * B].diag() + sim_matrix[2 * B:3 * B, 0:B].diag() +
                            sim_matrix[B:2 * B, 2 * B:3 * B].diag() + sim_matrix[2 * B:3 * B, B:2 * B].diag() +
                            sim_matrix[0:B, 3 * B:].diag() + sim_matrix[3 * B:, 0:B].diag() +
                            sim_matrix[B:2 * B, 3 * B:].diag() + sim_matrix[3 * B:, B:2 * B].diag() +
                            sim_matrix[2 * B:3 * B, 3 * B:].diag() + sim_matrix[3 * B:, 2 * B:3 * B].diag()
                            ) / float(sim_matrix.size(0))
    else:
        raise Exception("Sorry, we can not compute contrastive loss value!")

    return loss
