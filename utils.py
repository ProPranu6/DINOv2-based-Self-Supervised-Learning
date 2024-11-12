import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import numpy as np
import PIL
import numpy as np
from skimage.morphology import medial_axis


def skeletonise(segimg):
    skel, distance = medial_axis((segimg*255).astype(np.uint8), return_distance=True)
    return skel, distance

def pick_image_from(dataset):
    img, tseg = [(img, tseg) for img, tseg in dataset.shuffle(8).take(1)][0]
    img, tseg = img.numpy(), tseg.numpy() #tf.maximum(tf.image.resize(img, (512, 512), 'nearest'), 0).numpy(), tf.maximum(tf.image.resize(tseg, (512, 512), 'nearest'), 0).numpy()
    return img, tseg.squeeze()

 
def segment_through(img, model):
    img = np.expand_dims(img, axis=0)
    img = np.array(PIL.Image.fromarray((model(img).numpy() >= optimal_threshold).squeeze()).filter(PIL.ImageFilter.MedianFilter(size=3)))
    return img



def plots_(primg, segimg, skel, distance, tseg, tskel, tdistance):    
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True, )
    ax = axes.ravel()

    ax[0].imshow(primg, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('processed image')

    ax[1].imshow(segimg, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('segmented image')

    ax[2].imshow(tseg, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('true segmented image')


    ax[3].imshow(skel, cmap='gray')
    ax[3].axis('off')
    ax[3].set_title('skeleton image')

    #ax[4].imshow(skel*distance, cmap='magma')
    cax = ax[4].imshow(skel * distance, cmap='magma')
    cbar = fig.colorbar(cax, ax=ax[4], shrink=0.5)
    cbar.set_label('width')
    ax[4].axis('off')
    ax[4].set_title('width emph. skeleton image')

    ax[5].imshow(skel*distance, cmap='magma')
    ax[5].contour(segimg, [0.001], colors='w')
    ax[5].axis('off')
    ax[5].set_title('width emph. skeleton wt. contour image')

    ax[6].imshow(tskel, cmap='gray')
    ax[6].axis('off')
    ax[6].set_title('true skeleton image')

    #ax[4].imshow(skel*distance, cmap='magma')
    dax = ax[7].imshow(tskel * tdistance, cmap='magma')
    dbar = fig.colorbar(dax, ax=ax[7], shrink=0.5)
    dbar.set_label('width')
    ax[7].axis('off')
    ax[7].set_title('width emph. true skeleton image')

    ax[8].imshow(tskel*tdistance, cmap='magma')
    ax[8].contour(tseg, [0.001], colors='w')
    ax[8].axis('off')
    ax[8].set_title('width emph. true skeleton wt. contour image')

    plt.tight_layout()
    plt.show()
    return 



def compare_preds_on(dataset, model, batch_size=5, threshold=0.5, input_processors=[], model_processors=[]):
    datasubsetr = dataset.take(batch_size)
    for process in input_processors:
        datasubsetr = datasubsetr.map(process)

    datasubsetp = datasubsetr
    for process in model_processors:
        datasubsetp = datasubsetp.map(process)
    
    xrs, yrs = next(datasubsetr.batch(batch_size).as_numpy_iterator())
    xps, yps = next(datasubsetp.batch(batch_size).as_numpy_iterator())

    dataset_size = len(dataset)
    
    for bnumber in range(min(dataset_size, batch_size)):
        # Adjust figure size and DPI for better visualization
        fig, axs = plt.subplots(2, 4, figsize=(24, 12), dpi=150)
        
        xr, _ = np.expand_dims(xrs[bnumber], axis=0), np.expand_dims(yrs[bnumber], axis=0)
        xp, yp = np.expand_dims(xps[bnumber], axis=0), np.expand_dims(yps[bnumber], axis=0)
        
        pred = np.squeeze(model.predict(xp) >= threshold, axis=0)
        
        xp = np.squeeze(xp, axis=0)
        xr = np.squeeze(xr, axis=0)
        y = np.squeeze(yp, axis=0)
        xip = np.where(pred, np.full_like(xr, 255), xr)
        xit = np.where(y, np.full_like(xr, 255), xr)

        skel, distance = skeletonise(np.squeeze(pred, axis=-1))  # Skeletonize the prediction
        
        axs[0, 0].set_title('Raw Input')
        axs[0, 0].imshow(xr, cmap='gray')

        axs[0, 1].set_title('Processed Input')
        axs[0, 1].imshow(xp[:, :, 0:1], cmap='gray')

        axs[0, 2].set_title('Predicted')
        axs[0, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
       
        axs[0, 3].set_title('True')
        axs[0, 3].imshow(y, cmap='gray')

        axs[1, 0].set_title('Input + Predicted')
        axs[1, 0].imshow(xip, cmap='gray')

        axs[1, 1].set_title('Input + True')
        axs[1, 1].imshow(xit, cmap='gray')

        # Skeleton image
        axs[1, 2].set_title('Skeleton Image')
        axs[1, 2].imshow(skel, cmap='gray')
        axs[1, 2].axis('off')

        # Width emphasis skeleton image
        cax = axs[1, 3].imshow(skel * distance, cmap='magma')
        cbar = fig.colorbar(cax, ax=axs[1, 3], fraction=0.046, pad=0.04)  # Adjust color bar size
        cbar.set_label('Width')
        axs[1, 3].axis('off')
        axs[1, 3].set_title('Width Emph. Skeleton Image')

        # Improve layout spacing
        plt.tight_layout()
        plt.show()



def find_lower_outliers_IQR(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    lower_outliers = df[df < (q1 - 1.5 * IQR)]
    return lower_outliers


def find_normal_samples(df):
    mean = df.mean()
    std = df.std()
    lower_bound = mean - std
    upper_bound = mean + std
    
    # Filter samples within one standard deviation from the mean
    normal_samples = df[(df >= lower_bound) & (df <= upper_bound)]
    return normal_samples