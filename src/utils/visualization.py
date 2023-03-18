import wandb
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def wandb_pred(model, test_loader, device):
    # create a wandb Artifact to version each test step separately
    test_data_at = wandb.Artifact(
        "test_samples_" + str(wandb.run.id), type="Transformations"
    )
    # create a wandb.Table() in which to store predictions for each test step
    columns = ["Grid in", "Grid out"]
    test_table = wandb.Table(columns=columns)

    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        # for batch, (data,_) in enumerate(test_loader):
        # data = data.to(device)
        input_tensor = data.cpu()
        #transformed_input_tensor = model.stn(data).cpu()
        transformed_input_tensor = model.stn(data)
        transformed_input_tensor = transformed_input_tensor.cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor[:16]))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor[:16])
        )

        # Plot the results side-by-side
        test_table.add_data(wandb.Image(in_grid), wandb.Image(out_grid))

    # log predictions table to wandb
    test_data_at.add(test_table, "Transformations")
    wandb.run.log_artifact(test_data_at)



def visualise_stn(subnet_samples,uncertain_images,data,model,device,name='_',n_samples=5):
    
    assert len(subnet_samples)>=n_samples,'The number of samples should be smaller or equal to samples'
    
    stn = []
    if len(uncertain_images)==0:

        image_idx = list(range(0,n_samples))
        for i in image_idx:
            stn.append(data[i])
    else:
        for i in uncertain_images[:n_samples]:
            stn.append(data[i])


    f, axarr = plt.subplots(n_samples, 1,figsize=(15,15))
    #data  = next(iter(test_loader))[0]
    

    cat = torch.cat(stn)
    plot_data = cat.unsqueeze(0).permute(1,0,2,3)
    #breakpoint()
    input_tensor = data[:n_samples].cpu() #change to plot_data if checking uncertain pics [n_samples:16]

    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor))

    mean = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    with torch.no_grad():
        for i,sample in enumerate(subnet_samples[:n_samples]):
            torch.nn.utils.vector_to_parameters(sample, model.parameters())
            transformed_input_tensor = model.stn(data[:n_samples].to(device)) #change to plot_data if checking uncertain pics
            #torch.nn.utils.vector_to_parameters(mean,model.parameters())
            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor.cpu()))
        #Plot the results side-by-side
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[i].imshow(out_grid)
            axarr[i].set_title(f'Transformed Images: sample {i}')
            axarr[i].set_axis_off()
            
            plt.subplots_adjust(left=0.5, bottom=1, right=1.5, top=2, wspace=0.2, hspace=0.5)
            plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/{name}STN_images.png',bbox_inches='tight')
            wandb.Image(plt, caption ="Samples")
        f.tight_layout()




def visualise_samples(uncertain_images,dataloader,data,labels,preds,images_la,images,n_samples=5,plotype='uncertainties'):
    
    if len(uncertain_images)==0:

        sample = list(range(0,n_samples))
    else:
        sample = uncertain_images[:n_samples]

    
    for i in range(len(sample)):
        fig,ax = plt.subplots(1,4, figsize=(20,3))
        img = data[sample[i]]#.permute(1,2,0)
        img = convert_image_np(img)
        label = labels[sample[i]]

        #plots
        
        ax[0].imshow(img)

        #bar plot
        for j in range(0,preds.shape[0]):
            ax[1].scatter(range(preds.shape[2]), preds[j][sample[i]].cpu().detach().numpy(),alpha=0.1,color = 'gray')
            ax[1].set_ylim([0, 1])

        # whisker plots    
        # red_circle = dict(markerfacecolor='red', marker='o')
        # mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='green')
        # ax[1].boxplot(preds[:,sample[i],:].cpu().detach().numpy(),vert = True, flierprops =red_circle,showmeans=True, meanprops=mean_shape)
        # ax[1].set_xticklabels(['0', '1','2', '3','4','5','6','7','8','9'])

        #ax[2].bar(range(10),images_la[sample[i]].detach().numpy())
        ax[2].bar(range(preds.shape[2]),preds[:,sample[i]].mean(axis=0).cpu().detach().numpy())
        

        ax[2].set_ylim([0,1])

        ax[3].bar(range(10),images[sample[i]].detach().numpy())
        ax[3].set_ylim([0,1])

        #titles
        ax[0].set_title(f'Input image {label}',size=16)
        ax[1].set_title('Laplace posterior samples',size=16)
        ax[2].set_title('Laplace posterior predictive probabilities',size=16)
        ax[3].set_title('MAP prediction probabilities',size=16)


        ax[1].set_ylabel('Probabilities',size=15)
        ax[2].set_ylabel('Probabilities',size=15)
        ax[3].set_ylabel('Probabilities',size=15)

        ax[1].set_xlabel('Classes',size=15)
        ax[2].set_xlabel('Classes',size=15)
        ax[3].set_xlabel('Classes',size=15)
        ax[1].tick_params(axis='both', labelsize=13)
        ax[2].tick_params(axis='both', labelsize=13)
        ax[3].tick_params(axis='both', labelsize=13)


        
        plt.subplots_adjust(left=0.5, bottom=1, right=1.5, top=2, wspace=0.5, hspace=0.5)
        plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/{plotype} {i}.png',bbox_inches='tight')
        wandb.Image(plt, caption =f"{plotype}")

        
        #if plotype=='uncertainties':
            
        #    plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/{plotype} {i}.png',bbox_inches='tight')
            
        
        #elif plotype == 'misclassification':

        #    plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/misClassified_images {i}.png',bbox_inches='tight')
        
        #else:
        #    plt.savefig(f'/zhome/fc/5/104708/Desktop/Thesis/src/visualization/OOD_images {i}.png',bbox_inches='tight')
        

    #wandb.log({f"{plotype}":fig})
    fig.tight_layout()