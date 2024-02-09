## With the help of chatgpt I learnt how to compute the loss for the generator model based on this paper: https://arxiv.org/pdf/2109.13354.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

BCE_sum = nn.BCELoss(reduction='sum')
sigmoid = nn.Sigmoid()
def reconstruction_loss(x, x_hat):
    # Binary Cross-Entropy Loss for reconstruction
    bce_loss_sum = BCE_sum(x_hat, x)
    return bce_loss_sum

def kl_divergence_loss(mean, logvar):
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl_loss

def vaegan_loss(x, mean, logvar, lambda_reg=1.0, alpha=1.0):
    # Sigmoid
    mean, logvar = sigmoid(mean), sigmoid(logvar)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(x, torch.ones_like(x))

    # Regularization Term (KL Divergence)
    reg_loss = kl_divergence_loss(mean, logvar)


    # Total Loss
    total_loss = (alpha * recon_loss) + (lambda_reg * reg_loss)

    return total_loss