import pickle
from audioop import add
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from scipy.stats import entropy
import matplotlib.ticker as mtick
import math
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)   

def main():

    plot_subset()
    # objects_laplace = get_object('/zhome/fc/5/104708/Desktop/Thesis/Polas_exp_KMNIST_variance_Laplace.p')
    # theta_one_lap = objects_laplace['one']
    # theta_two_lap = objects_laplace['two']
    # theta_three_lap = objects_laplace['three']
    # theta_four_lap = objects_laplace['four']
    # theta_five_lap = objects_laplace['five']
    # theta_five_lap = objects_laplace['six']



    #kmnist_lap = get_object('/zhome/fc/5/104708/Desktop/Thesis/Polas_exp_KMNIST_variance_Laplace.p')
    # mnist_lap  = get_object('/zhome/fc/5/104708/Desktop/Thesis/Polas_exp_MNIST_variance_Laplace.p')
    # kmnist_pstn_betas = get_object('/zhome/fc/5/104708/Desktop/pSTN-baselines/experiments/11_07_MNIST/theta_stats/d=KMNIST-m=pstn-p=6-fold=0-kl=weight_kl_3e-05-betaP=1-lr=0.001-lrloc=0.1-varinit=-20.0/test_beta.p')
    # mnist_pstn_betas  = get_object('/zhome/fc/5/104708/Desktop/pSTN-baselines/experiments/11_07_MNIST/theta_stats/d=MNIST-m=pstn-p=6-fold=0-kl=weight_kl_3e-05-betaP=1-lr=0.001-lrloc=0.1-varinit=-20.0/test_beta.p')
    #kmnist_pstn_samples = get_object('/zhome/fc/5/104708/Desktop/pSTN-baselines/experiments/13_07_MNIST/UQ/d=KMNIST-m=pstn-p=6-fold=0-kl=weight_kl_3e-05-betaP=1-lr=0.001-lrloc=0.1-varinit=-20.0/UQ_results.p')
    #mnist_pstn_samples = get_object('/zhome/fc/5/104708/Desktop/pSTN-baselines/experiments/13_07_MNIST/UQ/d=MNIST-m=pstn-p=6-fold=0-kl=weight_kl_3e-05-betaP=1-lr=0.001-lrloc=0.1-varinit=-20.0/UQ_results.p')

    
    #transformed_kmnist = kmnist_pstn_samples['transformed']
    #transformed_mnist = mnist_pstn_samples['transformed']
    #transfo_kmnist = transformed_kmnist.reshape([transformed_kmnist.shape[0]*transformed_kmnist.shape[2],transformed_kmnist.shape[1],transformed_kmnist.shape[3]])
    #transfo_mnist = transformed_mnist.reshape([transformed_mnist.shape[0]*transformed_mnist.shape[2],transformed_mnist.shape[1],transformed_mnist.shape[3]])

    #th_one, th_two, th_three, th_four, th_five, th_six = theta_variance(transfo_kmnist)
    # transformed_mnist = mnist_pstn_samples['transformed']
    # transfo_mnist = np.vstack(transformed_mnist)
    # variances_pstn_mnist = [np.var(transfo_mnist[i]) for i in range(len(transfo_mnist))]
    # transfo_kmnist = np.vstack(transformed_kmnist)
    # variances_pstn_kmnist = [np.var(transfo_kmnist[i]) for i in range(len(transfo_kmnist))]
    
    # kmnist_pstn = np.vstack(kmnist_pstn_betas)
    # mnist_pstn = np.vstack(mnist_pstn_betas)
    #breakpoint()
    
    #bar plot
    # f, ax = plt.subplots(2, 3,figsize=(20,20))
    # ax[0,0].hist(theta_one_lap, bins = 20, facecolor = 'blue',label = 'Laplace Th_one',alpha=0.5)
    # ax[0,1].hist(theta_two_lap, bins = 20, facecolor = 'blue',label = 'Laplace Th_two',alpha=0.5)
    # ax[0,2].hist(theta_three_lap, bins = 20, facecolor='blue',label = 'Laplace Th_three',alpha=0.5)
    # ax[1,0].hist(theta_four_lap, bins = 20, facecolor='blue',label = 'Laplace Th_four',alpha=0.5)
    # ax[1,1].hist(theta_five_lap, bins = 20, facecolor='blue',label = 'Laplace Th_five',alpha=0.5)
    # ax[1,2].hist(theta_five_lap, bins = 20, facecolor='blue',label = 'Laplace Th_six',alpha=0.5)

    # ax[0,0].hist(np.clip(th_one,0,0.30), bins = 20, facecolor = 'orange',label = 'pSTN Th_one',alpha=0.5)
    # ax[0,1].hist(np.clip(th_two,0,0.40), bins = 20, facecolor = 'orange',label = 'pSTN Th_two',alpha=0.5)
    # ax[0,2].hist(np.clip(th_three,0,0.04), bins = 20, facecolor='orange',label = 'pSTN Th_three',alpha=0.5)
    # ax[1,0].hist(np.clip(th_four,0,0.20), bins = 20, facecolor='orange',label = 'pSTN Th_four',alpha=0.5)
    # ax[1,1].hist(np.clip(th_five,0,0.04), bins = 20, facecolor='orange',label = 'pSTN Th_five',alpha=0.5)
    # ax[1,2].hist(np.clip(th_six,0,0.04), bins = 20, facecolor='orange',label = 'pSTN Th_six',alpha=0.5)
    # xlim = [0,1]
    # ax[0,0].set_xlim(xlim)
    # ax[0,1].set_xlim(xlim)
    # ax[0,2].set_xlim(xlim)
    # ax[1,0].set_xlim(xlim)
    # ax[1,1].set_xlim(xlim)
    # ax[1,2].set_xlim(xlim)
    
    
    # ax[0,0].axvline(np.mean(mnist_lap), color='orange', linestyle='dashed', linewidth=1)
    # ax[0,1].axvline(np.mean(mnist_pstn), color='blue', linestyle='dashed', linewidth=1)
    # ax[0,2].axvline(np.mean(variances_pstn_mnist), color='blue', linestyle='dashed', linewidth=1)
    # ax[1,0].axvline(np.mean(kmnist_lap), color='orange', linestyle='dashed', linewidth=1)
    # ax[1,1].axvline(np.mean(kmnist_pstn), color='blue', linestyle='dashed', linewidth=1)
    # ax[1,2].axvline(np.mean(variances_pstn_kmnist), color='blue', linestyle='dashed', linewidth=1)


    # ax[0,0].legend(loc='upper right')
    # ax[0,1].legend(loc='upper right')
    # ax[0,2].legend(loc='upper right')
    # ax[1,0].legend(loc='upper right')
    # ax[1,1].legend(loc='upper right')
    # ax[1,2].legend(loc='upper right')

    # plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/loc_varianceKPSTN.png',bbox_inches='tight')
    # f.tight_layout()




def get_object(filepath):
    #objects = []
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                #objects.append(pickle.load(openfile))
                objects = pickle.load(openfile)
            except EOFError:
                break
    return objects



def theta_variance(theta):
    
    theta_one = []
    theta_two = []
    theta_three = []
    theta_four = []
    theta_five = []
    theta_six = []
    

    for image in range(theta.shape[0]):
            
        theta_one.append(theta[image][:10][:,0])
        theta_two.append(theta[image][:10][:,1])
        theta_three.append(theta[image][:10][:,2])
        theta_four.append(theta[image][:10][:,3])
        theta_five.append(theta[image][:10][:,4])
        theta_six.append(theta[image][:10][:,5])
    

    theta_one_var = [np.var(theta_one[i]) for i in range(len(theta_one))]
    theta_two_var = [np.var(theta_two[i]) for i in range(len(theta_two))]
    theta_three_var = [np.var(theta_three[i]) for i in range(len(theta_three))]
    theta_four_var = [np.var(theta_four[i]) for i in range(len(theta_four))]
    theta_five_var = [np.var(theta_five[i]) for i in range(len(theta_five))]
    theta_six_var = [np.var(theta_six[i]) for i in range(len(theta_six))]

    
    return theta_one_var,theta_two_var,theta_three_var,theta_four_var,theta_five_var,theta_six_var



def plot_subset():

    fig,ax = plt.subplots(1,1,figsize=(10,10))

    map_acc = [84.90, 93.4, 97.10, 98.30, 98.80]
    lap_acc = [89.95, 95.56, 97.74, 98.74, 99.07]

    map_error = [0.05, 0.11, 0.12, 0.06, 0.02]
    lap_error = [0.65, 0.38, 0.21, 0.13, 0.03]

    subsets = ['MNIST 100','MNIST 500','MNIST 2000','MNIST 5000','MNIST 10000']
    

    #plots
    ax.errorbar(subsets,map_acc,map_error,color = 'g',marker='^',label = 'MAP')
    ax.errorbar(subsets,lap_acc,lap_error,color='b',marker='o',label = 'Laplace Optimal')
    
    # acc = ax.get_yticks()
    # ax.set_yticklabels(['{:,.0%}'.format(x) for x in acc])
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_ylabel('Test Accuracy')
    #titles
    ax.set_title('MNIST Subsets')
    


    #legends
    ax.legend()
    

    #xlabels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    
    labels[0] = 'MNIST 100'
    labels[1] = 'MNIST 500'
    labels[2] = 'MNIST 2000'
    labels[3] = 'MNIST 5000'
    labels[4] = 'MNIST 10000'
   
    ax.set_xticklabels(labels)
    
    plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/subsetacc.png',bbox_inches='tight')
    fig.tight_layout()





if __name__ =='__main__':
    main()