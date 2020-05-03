""" Learned params idk why"""


def get_device(gpu):
    import torch
    torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

