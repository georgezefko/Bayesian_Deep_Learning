import warnings

warnings.simplefilter("ignore", UserWarning)

from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
import torch
from laplace import Laplace, marglik_training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.distributions as dists
from netcal.metrics import ECE
from src.data import make_dataset
from src.utils import reliability_diagram
from src.utils import visualization
from laplace.utils import ModuleNameSubnetMask
import wandb
from laplace.utils import LargestMagnitudeSubnetMask
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

import math
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)    

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def uncertainties(dataloader,images_la,images):
    uncertain_images = []
    for i in range(len(dataloader.dataset)):
        vla,_ = torch.max(images_la[i],-1)
        v,_= torch.max(images[i],-1)
        if abs(vla-v)>0.1:
            uncertain_images.append(i)
    return uncertain_images

def laplace(model,dataloader,method='last',module='fc2'):

    if method == 'last':
        la = Laplace(
            model,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure="kron",
            
        )
        la.fit(dataloader)
        # la.optimize_prior_precision(method="marglik")
        #la.prior_precision = torch.tensor([0.001])
        subnetwork_indices=0

    elif method=='diag':
        la = Laplace(
            model,
            "classification",
            subset_of_weights="all",
            hessian_structure="diag",
        )
        la.fit(dataloader)
        # la.optimize_prior_precision(method="marglik")
        #la.prior_precision = torch.tensor([0.001])
        subnetwork_indices=0

    else:
        subnetwork_mask = ModuleNameSubnetMask(model, module_names=module)
        subnetwork_mask.select()
        subnetwork_indices = subnetwork_mask.indices
    
        la = Laplace(
            model,
            "classification",
            subset_of_weights="subnetwork",
            hessian_structure="diag",
            subnetwork_indices = subnetwork_indices.type(torch.LongTensor),
            
        )
        la.fit(dataloader)

    return la,subnetwork_indices

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []
    target = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for x, t in dataloader:
        x,t = x.to(device),t.to(device)
        target.append(t)
        if laplace:
            py.append(model(x, pred_type='nn'))#,link_approx='probit',diagonal_output=True))
        else:
            py.append(torch.softmax(model(x), dim=-1))
    
    

    images = torch.cat(py).cpu()
    labels =torch.cat(target, dim=0).cpu()
   
   
    # y_conf = images.cpu().detach().numpy() 
    # y_conf_la = images.cpu().detach().numpy()
    # labels_encoded = torch.nn.functional.one_hot(labels, num_classes=10)
    # diagram = reliability_diagram.diagrams(y_conf,y_conf_la,labels_encoded,title='MAP Realiability full')
    # ece_map,_ = diagram.get_metrics(y_conf_la,labels_encoded)

    acc_map = (images.argmax(-1) == labels).float().mean()
    ece_map = ECE(bins=15).measure(images.numpy(), labels.numpy())
    nll_map = -dists.Categorical(images).log_prob(labels).mean()
    
    return acc_map,ece_map,nll_map,images,labels

def sanity_check_laplace(la,sublaplace,dataloader,subnetwork_indices):

    priors = torch.tensor([0.001,10000])
    state = ['no_opt','prior','default']

    laplace_samples_default = []
    sublaplace_samples_default = []
    laplace_sample_priors = []
    sublaplace_sample_priors = []
    laplace_sample_no_opt = []
    #sublaplace_sample_no_opt = []
    la_acc=[]
    la_ece = []
    la_nll =[]
    subla_acc=[]
    subla_ece = []
    subla_nll =[]
    la_metrics = []
    sub_metrics=[]
    
    for i in range(len(state)):
        print(state[i])

        if state[i]=='no_opt':

            laplace = la.sample(n_samples=10)
            laplace_sample_no_opt.append(laplace)

            fullset = sublaplace.sample(n_samples=10)
            subset = fullset[:,subnetwork_indices]
            sublaplace_samples_default.append(subset)

            acc_la,ece_la,nll_la,_,_ = predict(dataloader, la, laplace=True)
            acc_sublaplace,ece_sublaplace,nll_sublaplace,_,_ = predict(dataloader, sublaplace, laplace=True)
            la_acc.append(acc_la)
            la_ece.append(ece_la)
            la_nll.append(nll_la)
            subla_acc.append(acc_sublaplace)
            subla_ece.append(ece_sublaplace)
            subla_nll.append(nll_sublaplace)


        elif state[i]=='prior':
            for j in range(len(priors)):

                la.prior_precision = priors[j]
                sublaplace.prior_precision = priors[j]

                laplace = la.sample(n_samples=10)
                laplace_sample_priors.append(laplace)

                fullset = sublaplace.sample(n_samples=10)
                subset = fullset[:,subnetwork_indices]
                sublaplace_sample_priors.append(subset)
                acc_la,ece_la,nll_la,_,_ = predict(dataloader, la, laplace=True)
                acc_sublaplace,ece_sublaplace,nll_sublaplace,_,_ = predict(dataloader, sublaplace, laplace=True)

                la_acc.append(acc_la)
                la_ece.append(ece_la)
                la_nll.append(nll_la)
                subla_acc.append(acc_sublaplace)
                subla_ece.append(ece_sublaplace)
                subla_nll.append(nll_sublaplace)

        elif state[i]=='default':

            la.optimize_prior_precision(method="marglik")
            laplace = la.sample(n_samples=10)
            laplace_samples_default.append(laplace)
            
            # sublaplace.optimize_prior_precision(method='marglik')
            # fullset = sublaplace.sample(n_samples=10)
            # subset = fullset[:,subnetwork_indices]
            # sublaplace_samples_default.append(subset)

            acc_la,ece_la,nll_la,_,_ = predict(dataloader, la, laplace=True)
            # acc_sublaplace,ece_sublaplace,nll_sublaplace,_,_ = predict(dataloader, sublaplace, laplace=True)
            la_acc.append(acc_la)
            la_ece.append(ece_la)
            la_nll.append(nll_la)
            # subla_acc.append(acc_sublaplace)
            # subla_ece.append(ece_sublaplace)
            # subla_nll.append(nll_sublaplace)
        else:
            pass
        
        
        la_metrics.append(la_acc)
        la_metrics.append(la_ece)
        la_metrics.append(la_nll)
        sub_metrics.append(subla_acc)
        sub_metrics.append(subla_ece)
        sub_metrics.append(subla_nll)

    return laplace_samples_default, sublaplace_samples_default,laplace_sample_priors ,sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics

def results(laplace_samples_default, sublaplace_samples_default,laplace_sample_priors ,sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics,module='fc2'):
    #laplace
    lamargilk = laplace_samples_default[0][0].cpu().detach().numpy()
    small = laplace_sample_priors[0][0].detach().cpu().numpy()
    large = laplace_sample_priors[1][0].detach().cpu().numpy()
    no_opt = laplace_sample_no_opt[0][0].detach().cpu().numpy()

    #subnetwork laplace
    subdefault = sublaplace_samples_default[0][0].cpu().detach().numpy()
    subsmall = sublaplace_sample_priors[0][0].cpu().detach().numpy()
    sublarge = sublaplace_sample_priors[1][0].cpu().detach().numpy()
    #subno_opt = sublaplace_sample_no_opt[0][0].detach().cpu().numpy()


    fig, axes = plt.subplots(nrows=2,ncols = 4 , sharey=False,figsize=(20,10))

    #laplace plots
    axes[0,0].hist(no_opt,bins=15,facecolor='blue', alpha=0.5)
    axes[0,1].hist(small,bins=15)
    axes[0,2].hist(large,bins=15)
    axes[0,3].hist(lamargilk,bins=15,facecolor='blue', alpha=0.8)


    #subnetwrok laplace
    axes[1,0].hist(subdefault,bins=15,facecolor='blue', alpha=0.5)
    axes[1,1].hist(subsmall,bins=15)
    axes[1,2].hist(sublarge,bins=15)
    #axes[1,3].hist(subdefault,bins=15)

    axes[0,1].set_ylabel('No of Samples', fontsize='medium') 
    axes[1,1].set_ylabel('No of Samples', fontsize='medium') 
    axes[1,1].set_xlabel('Samples', fontsize='medium')
    axes[1,2].set_xlabel('Samples', fontsize='medium')



    # axes[0,0].set_title(f'Laplace no prior optimization \n acc {la_metrics[0][0]:.1%}, ece {la_metrics[1][0]:.1%}, NLL {la_metrics[2][0]:.3}',size=15)
    # axes[0,1].set_title(f'laplace last layer prior precision 0.01 \n acc {la_metrics[0][1]:.1%}, ece {la_metrics[1][1]:.1%}, NLL {la_metrics[2][1]:.3}',size=15)
    # axes[0,2].set_title(f'laplace last layer prior precision 1000 \n acc {la_metrics[0][2]:.1%}, ece {la_metrics[1][2]:.1%}, NLL {la_metrics[2][2]:.3}',size=15)
    # axes[0,3].set_title(f'Laplace prior optimization \n acc {la_metrics[0][3]:.1%}, ece {la_metrics[1][3]:.1%}, NLL {la_metrics[2][3]:.3}',size=15)

    axes[0,1].set_title(f'LLLaplace prior precision 0.01',size=15)
    axes[0,2].set_title(f'LLLaplace prior precision 1000',size=15)
    axes[1,1].set_title(f'Subnetwork laplace (STN 2nd last layer) \n prior precision 0.01',size=15)
    axes[1,2].set_title(f'Subnetwork laplace (STN 2nd last layer) \n prior precision 1000',size=15)
    # axes[1,0].set_title(f'Subnetwork Laplace default \n acc {sub_metrics[0][0]:.1%}, ece {sub_metrics[1][0]:.1%}, NLL {sub_metrics[2][0]:.3}',size=15)
    # axes[1,1].set_title(f'Subnetwork laplace {module} \n prior precision 0.01 \n acc {sub_metrics[0][1]:.1%}, ece {sub_metrics[1][1]:.1%}, NLL {sub_metrics[2][1]:.3}',size=15)
    # axes[1,2].set_title(f'Subnetwork laplace {module} \n prior precision 1000 \n acc {sub_metrics[0][2]:.1%}, ece {sub_metrics[1][2]:.1%}, NLL {sub_metrics[2][2]:.3}',size=15)
    #axes[1,3].set_title(f'Subnetwork laplace {module} \n prior optimization \n acc {sub_metrics[0][3]:.1%}, ece {sub_metrics[1][3]:.1%}, NLL {sub_metrics[2][3]:.3}',size=15)
    
    fig.delaxes(axes[1][3])
    plt.subplots_adjust(left=0.5, bottom=1, right=1.5, top=2, wspace=0.2, hspace=0.5)
    plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/STN_lastlayer_results_MNIST.png',bbox_inches='tight')
    wandb.Image(fig,caption="STN results")


def sample_predictions(ma,lap,device,dataloader):
    with torch.no_grad():
        py_la = []
        py=[]
        target_la = []
        target=[]
        test_image = []
        #temp= []
        for x, t in dataloader:
            x,t = x.to(device),t.to(device)
            target.append(t)
            test_image.append(x)
            target_la.append(t)
            py_la.append(lap(x,pred_type='glm'))
            py.append(torch.softmax(ma(x), dim=-1))

        images = torch.cat(py).cpu()
        labels =torch.cat(target, dim=0).cpu()
        images_la = torch.cat(py_la).cpu()
        labels_la =torch.cat(target_la, dim=0).cpu()
        return images,labels,images_la,labels_la,test_image

def grid_cv(sub_laplace,test_loader):
            
    
        #GridCv search on optimal prior precision
        subnetwork_acc =[]
        subnetwork_ece = []
        subnetwork_nll =[]
        prior_pre = torch.tensor([250,260,270,280,290,300])
        #prior_pre = torch.tensor([10, 50, 100, 200,500])
        
        for i in range(len(prior_pre)):
            
            sub_laplace.prior_precision = prior_pre[i]
            acc_sub,ece_sub,nll_sub,_,_ = predict(test_loader,sub_laplace,laplace=True)
            
            subnetwork_acc.append(acc_sub)
            subnetwork_ece.append(ece_sub)
            subnetwork_nll.append(nll_sub)
            print(
                f"[Sub Network Laplace] prior: {prior_pre[i]} Acc.: {acc_sub:.1%}; ECE: {ece_sub:.1%}; NLL: {nll_sub:.3}"
            )
        idx_acc = subnetwork_acc.index(max(subnetwork_acc))
        idx_nll = subnetwork_nll.index(min(subnetwork_nll))
        idx_ece = subnetwork_ece.index(min(subnetwork_ece))

        
        if (idx_acc==idx_ece==idx_nll):
            return prior_pre[idx_acc]
        else:
            return prior_pre[idx_ece]
        

def subset_performance(model,device,batch_size,crop_size,dataset,misplacement,load,save,module='fc2',method='sub'):
        
        #Subsets ={'100':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_loc(512)_misMNIST_sub100_HPC.pth',
        #        '500':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_loc(512)_misMNIST_sub500_HPC.pth',
        #        '2K':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_loc(512)_misMNIST_sub2000_HPC.pth',
        #        '5K':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_loc(512)_misMNIST_sub5000_HPC.pth',
        #        '10K':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_loc(512)_misMNIST_sub10000_HPC.pth'
        #        }

        Subsets ={'100':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_MNIST_100_cheat_HPC.pth',
                '500':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_MNIST_500_cheat_HPC.pth',
                '2K':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_MNIST_2k_cheat_HPC.pth',
                '5K':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_MNIST_5k_cheat_HPC.pth',
                '10K':'/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_MNIST_10k_cheat_HPC.pth'
                }

        map_acc = []
        map_nll = []
        map_ece =[]
        subnetwork_acc = []
        subnetwork_nll = []
        subnetwork_ece = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for keys,values in Subsets.items():

            if keys == '100':

                print(keys)
                train_subset = 10

            elif keys == '500':

                print(keys)
                train_subset = 50

            elif keys == '2K':

                print(keys)
                train_subset = 200

            elif keys == '5K':

                print(keys)
                train_subset = 500

            elif keys == '10K':

                print(keys)
                train_subset = 1000

            model_path = values
            checkpoint = torch.load(model_path, map_location=device)
            # initialize state_dict from checkpoint to model
            model.load_state_dict(checkpoint["state_dict"],strict=False)
            train_loader,valid_loader,test_loader = make_dataset.data(batch_size,crop_size,train_subset,dataset,
                                                            misplacement,load,save,subset=True)
            acc_map,ece_map,nll_map,map_images,labels = predict(test_loader,model,laplace=False)
            sub_la,_ = laplace(model,train_loader,method = method,module=module)

            sub_la.optimize_prior_precision(method='marglik')
            print(sub_la.prior_precision)
            acc_sub,ece_sub,nll_sub,sub_la_images,_ = predict(test_loader,sub_la,laplace=True)

            print(
                    f"[Sub Network Laplace] Acc.: {acc_sub:.1%}; ECE: {ece_sub:.1%}; NLL: {nll_sub:.3}"
                )

            y_true = labels.cpu().detach().numpy()
            _,y_pred= torch.max(map_images,-1)
            y_conf = map_images.cpu().detach().numpy()

            _,y_pred_la = torch.max(sub_la_images,-1)
            y_conf_la = sub_la_images.cpu().detach().numpy()

            labels_encoded = torch.nn.functional.one_hot(labels, num_classes=10)

            # diagramL = ReliabilityDiagram(n_bins=15,metric = 'ECE')
            # diagramL.plot(map_images.numpy(), labels.numpy(),filename=f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/MAP Calibration{keys}.png')  # visualize miscalibration of uncalibrated
            # diagramL.plot(sub_la_images.numpy(), labels.numpy(),filename=f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/Laplace_calibration{keys}.png')   # visualize miscalibration of calibrated
            
            
            diagram = reliability_diagram.diagrams(y_conf,y_conf_la,labels_encoded,title=f'MAP Realiability Diagram {keys}')

            
            diagram.draw_reliability_graph(y_conf,labels_encoded,title=f'MAP Realiability Diagram {keys}')
            diagram.draw_reliability_graph(y_conf_la,labels_encoded,title=f'Laplace Realiability Diagram {keys}')
            diagram.draw_calibration_graph(y_conf,y_conf_la,labels_encoded,title=f'Calibration Curve{keys}')

            subnetwork_acc.append(acc_sub)
            subnetwork_ece.append(ece_sub)
            subnetwork_nll.append(nll_sub)
            
            map_acc.append(acc_map)
            map_nll.append(nll_map)
            map_ece.append(ece_map)
            if keys == '100':
                data = next(iter(test_loader))[0]
                uncertain_images = []#uncertainties(test_loader,sub_la_images,map_images)
                #breakpoint()
                samples = sub_la.sample(n_samples=100)
                visualization.visualise_stn(samples,uncertain_images,data,model,device,sub_la.prior_precision,n_samples=8)
            elif keys == '10K':
                data = next(iter(test_loader))[0]
                uncertain_images = []#uncertainties(test_loader,sub_la_images,map_images)
                #breakpoint()
                samples = sub_la.sample(n_samples=100)
                visualization.visualise_stn(samples,uncertain_images,data,model,device,sub_la.prior_precision,n_samples=8)
            
        '''
            if keys=='100':
                data = next(iter(test_loader))[0]

            #prior_pre = prior_pre 
        
            for j in range(len(prior_pre)):
            
                
                sub_la.prior_precision = prior_pre[j]
                acc_sub,ece_sub,nll_sub,sub_la_images,_ = predict(test_loader,sub_la,laplace=True)

                subnetwork_acc.append(acc_sub)
                subnetwork_ece.append(ece_sub)
                subnetwork_nll.append(nll_sub)

                if keys == '100':
                    uncertain_images = []#uncertainties(test_loader,sub_la_images,map_images)
                    #breakpoint()
                    samples = sub_la.sample(n_samples=100)
                    visualization.visualise_stn(samples,uncertain_images,data,model,device,prior_pre[j],n_samples=8)
        
        sub_acc = [subnetwork_acc[i:i+len(prior_pre)] for i in range(0, len(subnetwork_acc), len(prior_pre))]
        sub_ece = [subnetwork_ece[i:i+len(prior_pre)] for i in range(0, len(subnetwork_ece), len(prior_pre))]
        sub_nll = [subnetwork_nll[i:i+len(prior_pre)] for i in range(0, len(subnetwork_nll), len(prior_pre))]
        max_idx_acc = max([(max(sub_acc[i]),prior_pre[sub_acc[i].index(max(sub_acc[i]))]) for i in range(len(sub_acc))])
        min_idx_ece = min([(min(sub_ece[i]),prior_pre[sub_ece[i].index(min(sub_ece[i]))]) for i in range(len(sub_ece))])
        min_idx_nll = min([(min(sub_nll[i]),prior_pre[sub_nll[i].index(min(sub_nll[i]))]) for i in range(len(sub_nll))])

        if (max_idx_acc[1] == min_idx_ece[1]==min_idx_nll[1]):
            print('optimal prior precision along all metrics',max_idx_acc[1])
        else:
            print('optimal prior precision minimum ECE',min_idx_ece[1])
        
        N = len(Subsets)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.12       # the width of the bars

        metrics_sub = [subnetwork_acc,subnetwork_nll,subnetwork_ece]
        metrics_map = [map_acc,map_nll,map_ece]
        fig,ax = plt.subplots(1,3,figsize=(25,10))


        for j in range(len(metrics_sub)):
            
            yvals =[metrics_sub[j][i]  for i in range(0, len(metrics_sub[j]), len(prior_pre))]
            
            zvals = [metrics_sub[j][i+1]  for i in range(0, len(metrics_sub[j]), len(prior_pre))]
            kvals = [metrics_sub[j][i+2]  for i in range(0, len(metrics_sub[j]), len(prior_pre))]
            tvals = [metrics_sub[j][i+3]  for i in range(0, len(metrics_sub[j]), len(prior_pre))]
            mvals = [metrics_sub[j][i+4]  for i in range(0, len(metrics_sub[j]), len(prior_pre))]
            nvals = [metrics_sub[j][i+5]  for i in range(0, len(metrics_sub[j]), len(prior_pre))]


            pvals = [metrics_map[j][i]  for i in range(0, len(metrics_map[j]))]


            rects1 = ax[j].bar(ind, yvals, width,color='red',alpha=0.5)
            rects2 = ax[j].bar(ind+width, zvals, width, color='green')
            rects3 = ax[j].bar(ind+width*2, kvals, width, color='blue')
            rects4 = ax[j].bar(ind+width*3, tvals, width, color='magenta')
            rects5 = ax[j].bar(ind+width*4, mvals, width, color='yellow')
            rects6 = ax[j].bar(ind+width*5, nvals, width, color='grey')

            rects7 = ax[j].bar(ind+width*6, pvals, width, color='pink')

            acc = ax[0].get_yticks()
            ax[0].set_yticklabels(['{:,.0%}'.format(x) for x in acc])
            ece = ax[2].get_yticks()
            ax[2].set_yticklabels(['{:,.0%}'.format(x) for x in ece])


            ax[j].set_ylabel('Scores')
            ax[j].set_xticks(ind+width)
            ax[j].set_xticklabels( ('MNIST100', 'MNIST500', 'MNIST2000','MNIST5000','MNIST10000') )
            ax[j].legend( (rects1[0], rects2[0], rects3[0],rects4[0],rects5[0],rects6[0],rects7[0]), ('sigma 0.1', 'sigma 0.5', 'Default','Sigma 10','sigma20','Sigma 30','MAP') )
            # ax[j].legend( (rects1[0], rects1[1], rects1[2],rects1[3],rects1[4],rects4[0]), ('sigma 0.1', 'sigma 0.5', 'Default','Sigma 10','Sigma 30','MAP') )

            #titles
            ax[0].set_title(f'Accuracy for MNIST subsets{max_idx_acc[1]}')
            ax[1].set_title(f'NLL for MNIST subsets {min_idx_nll[1]}')
            ax[2].set_title(f'ECE for MNIST subsets {min_idx_ece[1]}')
        #plt.show()
        plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/Subset_Performance.png',bbox_inches='tight')
        image = wandb.Image(fig,caption="subsets performance") 
        wandb.log({'subset performance': image})
        
        '''
        subset_data = {
        'MAP 100':[f'{map_acc[0]:.1%}',f'{map_ece[0]:.1%}',f'{map_nll[0]:.3}'],
        'Laplace 100':[f'{subnetwork_acc[0]:.1%}',f'{subnetwork_ece[0]:.1%}',f'{subnetwork_nll[0]:.3}'],
        'MAP 500':[f'{map_acc[1]:.1%}',f'{map_ece[1]:.1%}',f'{map_nll[1]:.3}'],
        'Laplace 500':[f'{subnetwork_acc[1]:.1%}',f'{subnetwork_ece[1]:.1%}',f'{subnetwork_nll[1]:.3}'],
        'MAP 2K':[f'{map_acc[2]:.1%}',f'{map_ece[2]:.1%}',f'{map_nll[2]:.3}'],
        'Laplace 2K':[f'{subnetwork_acc[2]:.1%}',f'{subnetwork_ece[2]:.1%}',f'{subnetwork_nll[2]:.3}'],
        'MAP 5K':[f'{map_acc[3]:.1%}',f'{map_ece[3]:.1%}',f'{map_nll[3]:.3}'],
        'Laplace 5K':[f'{subnetwork_acc[3]:.1%}',f'{subnetwork_ece[3]:.1%}',f'{subnetwork_nll[3]:.3}'],
        'MAP 10K':[f'{map_acc[4]:.1%}',f'{map_ece[4]:.1%}',f'{map_nll[4]:.3}'],
        'Laplace 10K':[f'{subnetwork_acc[4]:.1%}',f'{subnetwork_ece[4]:.1%}',f'{subnetwork_nll[4]:.3}']
        }
        indices = ['Acc','ECE','NLL']
        laplace_report = pd.DataFrame(subset_data,index=indices)
        laplace_report.to_csv('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/Subsetreport.xslx')
        
        laplace_table = wandb.Table(data=laplace_report)
        wandb.log({'laplace_report':laplace_table})
        
        fig,ax = plt.subplots(1,3,figsize=(20,10))

        #plots
        ax[0].plot(map_acc,color = 'g',label = 'MAP')
        ax[0].plot(subnetwork_acc,color='b',label = 'Laplace Optimal')
        ax[1].plot(map_nll,color = 'g',label = 'MAP')
        ax[1].plot(subnetwork_nll,color='b',label = 'Laplace Optimal')
        ax[2].plot(map_ece,color = 'g',label = 'MAP')
        ax[2].plot(subnetwork_ece,color='b',label = 'Laplace Optimal')
        acc = ax[0].get_yticks()
        ax[0].set_yticklabels(['{:,.0%}'.format(x) for x in acc])
        ece = ax[2].get_yticks()
        ax[2].set_yticklabels(['{:,.0%}'.format(x) for x in ece])

        #titles
        ax[0].set_title('Accuracy')
        ax[1].set_title('NLL')
        ax[2].set_title('ECE')


        #legends
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        #xlabels
        labels = [item.get_text() for item in ax[0].get_xticklabels()]
        labels[1] = 'MNIST 100'
        labels[2] = 'MNIST 500'
        labels[3] = 'MNIST 2000'
        labels[4] = 'MNIST 5000'
        labels[5] = 'MNIST 10000'

        ax[0].set_xticklabels(labels,rotation=10)
        ax[1].set_xticklabels(labels,rotation=10)
        ax[2].set_xticklabels(labels,rotation=10)

       
        plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/Subset_Performance.png',bbox_inches='tight')
        wandb.Image(fig,caption="subsets performance") 
        
def ood_eval(model,sub_laplace,batch_size,crop_size,train_subset,device,dataset='misKMNIST',plotype='OOD',misplacement=True,load=True,save=True,subset=False):
        #out of distribution images
    if dataset== 'KMNIST':
        print('KMNIST')
        set = torchvision.datasets.KMNIST(root='src/data', train=False,
                                        download=True, transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ))
        loader = torch.utils.data.DataLoader(set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
        _,_,_,_,_,_ = ood_process(loader,model,sub_laplace,device,plotype =plotype,name='OODSamples')
                                           
    elif  dataset=='misKMNIST':
        print('misKMNIST')
        _,_,loader = make_dataset.data(batch_size,crop_size,train_subset,dataset,misplacement=True,load=True,save=True,subset=False)
        _,_,_,_,_,_ = ood_process(loader,model,sub_laplace,device,plotype =plotype,name='_')
    
    elif dataset== 'CIFAR':
        print('CFAR')
        set = torchvision.datasets.CIFAR10(root='src/data', train=False,
                                        download=True, transform=transforms.Compose(
                    [transforms.Resize((128, 128)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ))
        loader = torch.utils.data.DataLoader(set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
        _,_,_,_,_,_ = ood_process(loader,model,sub_laplace,device,plotype =plotype,name='OODSamples')
    
    else:
        print('Distribution Shift MNIST')
        degrees = [0,30,60,90,120,150,180]
        map_acc = []
        map_nll = []
        map_ece =[]
        subnetwork_acc = []
        subnetwork_nll = []
        subnetwork_ece = []

        for i  in degrees:
            print(f'Distribution shift rotation {i}')
            set = torchvision.datasets.MNIST(root='src/data', train=False,
                                            download=True, transform=transforms.Compose(
                        [transforms.ToTensor(), transforms.RandomRotation(degrees=(i)), transforms.Normalize((0.1307,), (0.3081,))]
                    ))
            loader = torch.utils.data.DataLoader(set, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
            #acc_map_ood,ece_map_ood,nll_map_ood,acc_sub_ood,ece_sub_ood,nll_sub_ood = ood_process(loader,model,sub_laplace,device,plotype =i,name=i)
        #_,_,_,_,_,_= ood_process(loader,model,sub_laplace,device,plotype ='rotated',name='random')
            plot_examples(loader,i)
        #     subnetwork_acc.append(acc_sub_ood)
        #     subnetwork_ece.append(ece_sub_ood)
        #     subnetwork_nll.append(nll_sub_ood)
            
        #     map_acc.append(acc_map_ood)
        #     map_nll.append(nll_map_ood)
        #     map_ece.append(ece_map_ood)
        
        # subset_data = {
        # 'MAP 30':[f'{map_acc[0]:.1%}',f'{map_ece[0]:.1%}',f'{map_nll[0]:.3}'],
        # 'Laplace 30':[f'{subnetwork_acc[0]:.1%}',f'{subnetwork_ece[0]:.1%}',f'{subnetwork_nll[0]:.3}'],
        # 'MAP 60':[f'{map_acc[1]:.1%}',f'{map_ece[1]:.1%}',f'{map_nll[1]:.3}'],
        # 'Laplace 60':[f'{subnetwork_acc[1]:.1%}',f'{subnetwork_ece[1]:.1%}',f'{subnetwork_nll[1]:.3}'],
        # 'MAP 90':[f'{map_acc[2]:.1%}',f'{map_ece[2]:.1%}',f'{map_nll[2]:.3}'],
        # 'Laplace 90':[f'{subnetwork_acc[2]:.1%}',f'{subnetwork_ece[2]:.1%}',f'{subnetwork_nll[2]:.3}'],
        # 'MAP 120':[f'{map_acc[3]:.1%}',f'{map_ece[3]:.1%}',f'{map_nll[3]:.3}'],
        # 'Laplace 120':[f'{subnetwork_acc[3]:.1%}',f'{subnetwork_ece[3]:.1%}',f'{subnetwork_nll[3]:.3}'],
        # 'MAP 150':[f'{map_acc[4]:.1%}',f'{map_ece[4]:.1%}',f'{map_nll[4]:.3}'],
        # 'Laplace 150':[f'{subnetwork_acc[4]:.1%}',f'{subnetwork_ece[4]:.1%}',f'{subnetwork_nll[4]:.3}'],
        # 'MAP 180':[f'{map_acc[5]:.1%}',f'{map_ece[5]:.1%}',f'{map_nll[5]:.3}'],
        # 'Laplace 180':[f'{subnetwork_acc[5]:.1%}',f'{subnetwork_ece[5]:.1%}',f'{subnetwork_nll[5]:.3}']
        # }
        # indices = ['Acc','ECE','NLL']
        # laplace_report = pd.DataFrame(subset_data,index=indices)
        # laplace_report.to_csv('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/RotateMNIST.xslx')
        
        # laplace_table = wandb.Table(data=laplace_report)
        # wandb.log({'laplace_report':laplace_table})
        
        # fig,ax = plt.subplots(1,3,figsize=(20,10))

        # #plots
        # ax[0].plot(map_acc,color = 'g',label = 'MAP')
        # ax[0].plot(subnetwork_acc,color='b',label = 'Laplace Optimal')
        # ax[1].plot(map_nll,color = 'g',label = 'MAP')
        # ax[1].plot(subnetwork_nll,color='b',label = 'Laplace Optimal')
        # ax[2].plot(map_ece,color = 'g',label = 'MAP')
        # ax[2].plot(subnetwork_ece,color='b',label = 'Laplace Optimal')
        # acc = ax[0].get_yticks()
        # ax[0].set_yticklabels(['{:,.0%}'.format(x) for x in acc])
        # ece = ax[2].get_yticks()
        # ax[2].set_yticklabels(['{:,.0%}'.format(x) for x in ece])

        # #titles
        # ax[0].set_title('Accuracy')
        # ax[1].set_title('NLL')
        # ax[2].set_title('ECE')


        # #legends
        # ax[0].legend()
        # ax[1].legend()
        # ax[2].legend()

        # #xlabels
        # labels = [item.get_text() for item in ax[0].get_xticklabels()]
        # labels[1] = '30'
        # labels[2] = '60'
        # labels[3] = '90'
        # labels[4] = '120'
        # labels[5] = '150'
        # labels[6] = '180'

        # ax[0].set_xticklabels(labels,rotation=10)
        # ax[1].set_xticklabels(labels,rotation=10)
        # ax[2].set_xticklabels(labels,rotation=10)
        # ax[0].set_xlabel('Rotation (degrees)')
        # ax[1].set_xlabel('Rotation (degrees)')
        # ax[2].set_xlabel('Rotation (degrees)')
        # plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/Rotated MNIST.png',bbox_inches='tight')


def ood_process(loader,model,sub_laplace,device,plotype,name):

    acc_map_ood,ece_map_ood,nll_map_ood,map_images_ood,labels_ood = predict(loader,model,laplace=False)

    print(
        f"[MAP] Acc.: {acc_map_ood:.1%}; ECE: {ece_map_ood:.1%}; NLL: {nll_map_ood:.3}"
        )

    acc_sub_ood,ece_sub_ood,nll_sub_ood,sub_la_images_ood,_ = predict(loader,sub_laplace,laplace=True)
    print(
            f"[Sub Network Laplace] Acc.: {acc_sub_ood:.1%}; ECE: {ece_sub_ood:.1%}; NLL: {nll_sub_ood:.3}"
        )

    ood_images = uncertainties(loader,sub_la_images_ood,map_images_ood)

    
    test_ood_images= []
    for x,_ in loader:
        x = x.to(device)
        test_ood_images.append(x)
        
    test_ood = torch.cat(test_ood_images)
    test_ood = test_ood[:50]

    preds_ood= sub_laplace.predictive_samples(test_ood,pred_type='nn',n_samples=100,diagonal_output=True)
    entropy_la = entropy_cal(preds_ood[:,:].mean(axis=0).cpu().detach().numpy())
    entropy_map = entropy_cal(map_images_ood.detach().numpy())
    avg_entropy_map = np.mean(entropy_map)
    avg_entropy_la = np.mean(entropy_la)
    print(f'Average entropy MAP {avg_entropy_map} \n Average entropy Laplace {avg_entropy_la}')
    subnet_samples = sub_laplace.sample(n_samples=100)
    
    ood_plot(subnet_samples,name)
    
    visualization.visualise_stn(subnet_samples,ood_images,test_ood.cpu(),model,device,name,n_samples=8)

    visualization.visualise_samples(ood_images,loader,test_ood.cpu(),labels_ood,preds_ood,sub_la_images_ood,map_images_ood,n_samples=3,plotype=plotype)
    return acc_map_ood,ece_map_ood,nll_map_ood,acc_sub_ood,ece_sub_ood,nll_sub_ood

    
def ood_plot(samples,name):
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.hist(samples.detach().cpu(),bins=20,facecolor='blue',alpha=0.1)
    plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/{name}oodsamples.png',bbox_inches='tight')

def plot_examples(data,name):
    images,_ = next(iter(data))
    img = images[2]
    img = visualization.convert_image_np(img)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(img)
    plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/{name} degrees.png',bbox_inches='tight')



def entropy_cal(array):

    total_entropy = []

    for i in array:
        
        #for j in i:
        total_entropy.append(entropy(i))
            # total_entropy = 0
            # total_entropy += -j * math.log(j, 2)
            # entropy.append(total_entropy)
    return total_entropy