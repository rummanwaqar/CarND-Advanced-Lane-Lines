import os
import matplotlib.pyplot as plt

# output image folder
output_folder = 'output_images/'

def display(images, labels, fname='', path=output_folder, figsize=None, cmap=None):
    assert len(images) > 0
    assert len(images) == len(labels)
    
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    
    for idx in range(len(images)):
        plt.subplot((len(images)//3)+1,3,idx+1)
        plt.title(labels[idx])
        if cmap is not None:
            plt.imshow(images[idx], cmap=cmap)
        else:
            plt.imshow(images[idx])
        
    if fname:
        plt.savefig(os.path.join(path, fname), bbox_inches='tight')
    plt.show()