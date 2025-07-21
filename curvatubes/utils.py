#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:03:21 2020

@author: Anna SONG

Auxiliary functions:
    - for file conversion: from nii.gz to np.array (and conversely)
    - for display: show slices of u
    - for initialization: u = u0 (random noise, balls, cubes)
        (not so useful if using the change of variable u = div A + m0 )
    - functions on u: double-well W(u), impose the average
    - obtain the coeffs (a20,a11,a02,b10,b01,c) from the non-reduced polynomial
        h2 (H-H0)^2 + k1 K + s (kap1 - kap1_0)^2 + t (kap2 - kap2_0)^2

"""

import numpy as np
import matplotlib.pyplot as plt
import re

import torch

# import nibabel as nib

def double_W(s):
    'Double-well centered at +1 and -1'
    return .25 * (1 - s ** 2) ** 2


def double_W_prime(s):
    'Derivative of the double-well'
    return s ** 3 - s  # = s * (s**2 - 1)


def manual_softplus(u, xi_bis=1e-6):
    '''Softplus approximation of the positive part u^+ with
    xi * log(1 + exp(u / xi)).
    Requires splitting wrt the sign of u for stable computations. '''
    return xi_bis + u * (u > 0) + xi_bis * torch.log(1 + torch.exp(- torch.abs(u) / xi_bis))


def save_nii(vol, filename, compressed=True):
    'Saves a 3D np.array into a .nii.gz or .nii file'
    # img = nib.Nifti1Image(vol, np.eye(4))
    # if compressed:
    #    img.to_filename(filename + '.nii.gz')
    # else:
    #    img.to_filename(filename + '.nii')

def slices(u, figsize=(12, 4), rescale=True, cmap='gray', save=False, title=''):
    '''Visualize three 2D slices of a 3D volume at z = Z//3, Z//2, or 2*Z//3
    rescale = True: grays between u.min and u.max
    rescale = False: grays between 0 and 1 (black/white beyond 0/1)'''
    if type(u) == torch.Tensor:
        u = u.detach().cpu().numpy()
    vmin = None;
    vmax = None
    Z = u.shape[0]
    if not rescale: vmin = 0.; vmax = 1.
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    ax[0].imshow(np.asarray(u[Z // 3], dtype=np.float64), vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1].imshow(np.asarray(u[Z // 2], dtype=np.float64), vmin=vmin, vmax=vmax, cmap=cmap)
    ax[2].imshow(np.asarray(u[2 * Z // 3], dtype=np.float64), vmin=vmin, vmax=vmax, cmap=cmap)
    fig.tight_layout()
    if title != '': fig.suptitle(title); fig.savefig(title + '.png')
    plt.show()


def single(img, figsize=(10, 10), rescale=True):
    'Visualize a single 2D image'
    if type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()
    vmin = None;
    vmax = None
    if not rescale: vmin = 0.; vmax = 1.
    plt.figure(figsize=figsize)
    plt.imshow(img, vmin=vmin, vmax=vmax, cmap='gray')
    plt.show()


def project_average(u, m=0):
    'Translates u in order to have [mean of u on the domain] = m'
    return u - u.mean() + m