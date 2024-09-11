import numpy as np
import pandas as pd
import torch.utils.data as data
from torch import nn, optim
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def CreatePlots(df, mach):
    mach_min = mach - 0.05
    mach_max = mach + 0.05
    df = df[df['MFJ'] < mach_max]
    df = df[df['MFJ'] > mach_min]

    pavg = 0.25 * (df['FAP1'].values + df['FAP2'].values +
                   df['FAP3'].values + df['FAP4'].values)
    Cp1n = (df['FAP1'].values-pavg)/(df['FAP5'].values-pavg)
    Cp2n = (df['FAP2'].values-pavg)/(df['FAP5'].values-pavg)
    Cp3n = (df['FAP3'].values-pavg)/(df['FAP5'].values-pavg)
    Cp4n = (df['FAP4'].values-pavg)/(df['FAP5'].values-pavg)
    Cpmn = (df['FAP5'].values-pavg)/df['FAP5'].values
    theta = df['YAW'].values
    phi = df['PITCH'].values
    # mach = df['MFJ'].values
    Pt = df['PTFJ'].values

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(df['FAP1'].values, theta, 'b.')
    axs[0].set_title('FAP1 vs Theta')
    axs[1].plot(df['FAP1'].values, phi, 'b.')
    axs[1].set_title('FAP1 vs Phi')
    plt.savefig('Figures/Mach={0} FAP1.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(df['FAP2'].values, theta, 'b.')
    axs[0].set_title('FAP2 vs Theta')
    axs[1].plot(df['FAP2'].values, phi, 'b.')
    axs[1].set_title('FAP2 vs Phi')
    plt.savefig('Figures/Mach={0} FAP2.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(df['FAP3'].values, theta, 'b.')
    axs[0].set_title('FAP3 vs Theta')
    axs[1].plot(df['FAP3'].values, phi, 'b.')
    axs[1].set_title('FAP3 vs Phi')
    plt.savefig('Figures/Mach={0} FAP3.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(df['FAP4'].values, theta, 'b.')
    axs[0].set_title('FAP4 vs Theta')
    axs[1].plot(df['FAP4'].values, phi, 'b.')
    axs[1].set_title('FAP4 vs Phi')
    plt.savefig('Figures/Mach={0} FAP4.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(df['FAP5'].values, theta, 'b.')
    axs[0].set_title('FAP5 vs Theta')
    axs[1].plot(df['FAP5'].values, phi, 'b.')
    axs[1].set_title('FAP5 vs Phi')
    plt.savefig('Figures/Mach={0} FAP5.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(theta, Cp1n, 'b.')
    axs[0].set_title('Cp1n vs. theta')
    axs[1].plot(phi, Cp1n, 'b.')
    axs[1].set_title('Cp1n vs. phi')
    plt.savefig('Figures/Mach={0} Cp1n.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(theta, Cp2n, 'b.')
    axs[0].set_title('Cp2n vs. theta')
    axs[1].plot(phi, Cp2n, 'b.')
    axs[1].set_title('Cp2n vs. phi')
    plt.savefig('Figures/Mach={0} Cp2n.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(theta, Cp3n, 'b.')
    axs[0].set_title('Cp3n vs. theta')
    axs[1].plot(phi, Cp3n, 'b.')
    axs[1].set_title('Cp3n vs. phi')
    plt.savefig('Figures/Mach={0} Cp3n.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(theta, Cp4n, 'b.')
    axs[0].set_title('Cp4n vs. theta')
    axs[1].plot(phi, Cp4n, 'b.')
    axs[1].set_title('Cp4n vs. phi')
    plt.savefig('Figures/Mach={0} Cp4n.png'.format(mach))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(theta, Cpmn, 'b.')
    axs[0].set_title('Cpmn vs. theta')
    axs[1].plot(phi, Cpmn, 'b.')
    axs[1].set_title('Cpmn vs. phi')
    plt.savefig('Figures/Mach={0} Cpmn.png'.format(mach))


if __name__ == "__main__":
    df = pd.read_csv('../../dataset/probe_cal.csv')
    CreatePlots(df, 0.2)
    CreatePlots(df, 0.3)
    CreatePlots(df, 0.4)
