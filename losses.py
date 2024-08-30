import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.nn.functional import mse_loss as mse_loss


def discriminator_loss(logits_real, logits_fake):
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss_pos = bce_loss(logits_real, torch.ones_like(logits_real))
    loss_neg = bce_loss(logits_fake, torch.zeros_like(logits_fake))
    loss = loss_pos + loss_neg
    
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    loss = None

    loss = bce_loss(logits_fake, torch.ones_like(logits_fake))

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    
    loss = None
    loss_pos = mse_loss(scores_real, torch.ones_like(scores_real))
    loss_neg = mse_loss(scores_fake, torch.zeros_like(scores_fake))
    loss = (loss_pos + loss_neg)*0.5
    
    return loss

def ls_generator_loss(scores_fake):
    loss = mse_loss(scores_fake, torch.ones_like(scores_fake))
    return loss

def wg_discriminator_loss(scores_real, scores_fake):
    real_loss = -torch.mean(scores_real)
    fake_loss = torch.mean(scores_fake)
    loss = (real_loss+fake_loss)
    
    return loss

def wg_generator_loss(scores_fake):
    loss = None
    loss = -torch.mean(scores_fake)
    return loss