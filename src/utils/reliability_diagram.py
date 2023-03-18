import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
from netcal.metrics import ECE


class diagrams:
    def __init__(self,preds,preds_la,labels,title):

        self.preds = preds
        self.preds_la = preds_la
        self.labels = labels
        self.title =title
    
    def calc_bins(self,preds,labels):
        # Assign each prediction to a bin
        num_bins = 10
        bins = np.linspace(0.1, 1, num_bins)
        binned = np.digitize(preds, bins)

        # Save the accuracy, confidence and size of each bin
        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)

        for bin in range(num_bins):
            bin_sizes[bin] = len(preds[binned == bin])
            if bin_sizes[bin] > 0:
                bin_accs[bin] = (labels[binned==bin]).sum() / bin_sizes[bin]
                bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

        return bins, binned, bin_accs, bin_confs, bin_sizes

    def get_metrics(self,preds,labels):
        ECE = 0
        MCE = 0
        bins, _, bin_accs, bin_confs, bin_sizes = self.calc_bins(preds,labels)

        for i in range(len(bins)):
            abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
            ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
            MCE = max(MCE, abs_conf_dif)

        return ECE, MCE
    

    def draw_reliability_graph(self,preds,labels,title):
        
        #ECE, MCE = self.get_metrics(preds,labels)
        # ece = ECE(bins=15).measure(preds, labels)
        bins, _, bin_accs, _, _ = self.calc_bins(preds,labels)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')

        # Error bars
        plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')


        # Draw bars and identity line
        plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
        #plt.errorbar(bins, bin_accs)

        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')

        #ECE and MCE legend
        # ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
        # MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
        # plt.legend(handles=[ECE_patch, MCE_patch])
        #plt.legend(handles=[ECE_patch])
        Output_patch = mpatches.Patch(color='blue', label='Output')
        GAP_patch = mpatches.Patch( alpha=0.3, color='r', hatch='\\', label='Gap')
        plt.legend(handles=[Output_patch, GAP_patch])
        plt.title(f'{title}')

        #wandb.Image(fig,caption="Reliability Graph")
        
        plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/{title}reliabilityplot.png',bbox_inches='tight')
        wandb.log({f'{title}reliability plot':plt})


    def draw_calibration_graph(self,preds,preds_la,labels,title):
        # ECE, MCE = self.get_metrics(preds,labels)
        # ECE_la, MCE_la = self.get_metrics(preds_la,labels)

        # ece = ECE(bins=15).measure(preds, labels)
        # ECE_la = ECE(bins=15).measure(preds_la, labels)



        bins, _, bin_accs, _, _ = self.calc_bins(preds,labels)
        bins_la, _, bin_accs_la, _, _ = self.calc_bins(preds_la,labels)


        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')

        # Error bars


        # Draw bars and identity line
        plt.errorbar(bins, bin_accs, uplims=True, lolims=True,color='b')
        plt.errorbar(bins_la, bin_accs_la, uplims=True, lolims=True,color='r')

        #plt.errorbar(bins, bin_accs)

        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')

        # ECE and MCE legend
        # ECE_patch = mpatches.Patch(color='blue', label='MAP ECE = {:.2f}%'.format(ECE*100))
        # MCE_patch = mpatches.Patch(color='blue', label='MAP MCE = {:.2f}%'.format(MCE*100))
        # ECE_patch_la = mpatches.Patch(color='red', label='Laplace ECE  = {:.2f}%'.format(ECE_la*100))
        # MCE_patch_la = mpatches.Patch(color='red', label='Laplace MCE = {:.2f}%'.format(MCE_la*100))
        # plt.legend(handles=[ECE_patch, MCE_patch,ECE_patch_la,MCE_patch_la])
        MAP_patch = mpatches.Patch(color='blue', label='MAP')
        LAP_patch = mpatches.Patch(color='red', label='Laplace')
        plt.legend(handles=[MAP_patch,LAP_patch])
        plt.title(f'{title}')

        
        plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/{title} calibrationgraph.png',bbox_inches='tight')
        wandb.log({f'{title}calibrationplot':plt})
        
  

