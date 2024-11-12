import tensorflow.keras.preprocessing.image as image_process
import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import cv2
from skimage.morphology import skeletonize

def pyfunc_parser(output_types):
    
    def actual_pyfunc_parser(func):
        def mapped_func(*args):

            outputs = tf.py_function(func, list(args), output_types)
            return outputs
        
        return mapped_func
    
    return actual_pyfunc_parser 


@pyfunc_parser(output_types=[tf.uint8, tf.uint8])
def mask_inputs(img, seg, mask=None):
    

    image_path, seg_path, mask_path = img.numpy(), seg.numpy(), mask.numpy() if type(mask) != type(None) else None
    
    img = image_process.load_img(image_path)
    seg = image_process.load_img(seg_path)
    
    if type(mask) != type(None):
        mask = image_process.load_img(mask_path)
    
    img = image_process.img_to_array(img)
    seg = image_process.img_to_array(seg)
    if type(mask) != type(None):
        mask = image_process.img_to_array(mask) // 255
    
    masked_img = (img*(mask if type(mask) != type(None) else 1)).astype(np.uint8)
    masked_seg = (seg*(mask if type(mask) != type(None) else 1)).astype(np.uint8)
    
    return masked_img, masked_seg

@pyfunc_parser(output_types=[tf.uint8, tf.uint8])
def resize_inputs(img, seg):
    
    #img = tf.maximum(tf.image.resize(img, (512, 512), 'nearest'), 0)
    #seg = tf.maximum(tf.image.resize(seg, (512, 512), 'nearest'), 0)
    
    #img = tf.cast(img, tf.uint8)
    #seg = tf.cast(seg, tf.uint8)
    img = tf.cast(tf.image.resize(img, (512, 512), 'nearest'), img.dtype)
    seg = tf.cast(tf.image.resize(seg, (512, 512), 'nearest'), seg.dtype)
    
    return img, seg

@pyfunc_parser(output_types=[tf.uint8, tf.uint8])
def extract_green_channel(img, seg):
    green_channel = img[:, :, 1]  
    green_channel_image = tf.expand_dims(green_channel, axis=-1) 

    green_channel = seg[:, :, 1]  
    green_channel_seg = tf.expand_dims(green_channel, axis=-1) 
    return green_channel_image, green_channel_seg

@pyfunc_parser(output_types=[tf.uint8, tf.uint8])
def clahe_gray(img, seg, clip_limit=2.0, grid_size=(8, 8), invertimg=True):
    clahe = cv2.createCLAHE(clipLimit=clip_limit.numpy(), tileGridSize=grid_size.numpy())
    clahe_image = np.expand_dims(clahe.apply(img.numpy()), -1) 
    if invertimg:
        clahe_image = 255 - clahe_image
    return clahe_image, seg.numpy()

@pyfunc_parser(output_types=[tf.float32, tf.uint8])
def label(img, seg):
    img, seg = img.numpy(), seg.numpy()
    seg = (seg>0.0)*1
    return img, seg

@pyfunc_parser(output_types=[tf.float32, tf.uint8])
def normalize(img, seg):
    img, seg = img.numpy(), seg.numpy()
    img = (img-np.mean(img))/(np.std(img)+1e-7)
    return img, seg

@pyfunc_parser(output_types=[tf.float32, tf.uint8])
def fill_cuts(img, seg):
    img, seg = img.numpy(), seg.numpy()
    seg_skeleton_ext = skeletonize(cv2.dilate(seg.astype(np.uint8), (5, 5), iterations=4)).astype(np.uint8)
    seg_enh =  np.expand_dims(cv2.bitwise_or(seg_skeleton_ext, seg), axis=-1)
    return img, seg_enh

@pyfunc_parser(output_types=[tf.uint8, tf.uint8])
def crop_center_square(img, seg):
    height, width = img.shape[:2]
    side_length = min(height, width)
    x = (width - side_length) // 2
    y = (height - side_length) // 2
    cropped_img = img[y:y+side_length, x:x+side_length]
    cropped_seg = seg[y:y+side_length, x:x+side_length]
    return cropped_img, cropped_seg

@pyfunc_parser(output_types=[tf.uint8, tf.uint8])
def crop_conj_region(img, seg, widthx=500, widthy=2500, heightx=500, heighty=1800):
    cropped_img = img[heightx:heighty, widthx:widthy]
    cropped_seg = seg[heightx:heighty, widthx:widthy]
    return cropped_img, cropped_seg

import tensorflow as tf
import numpy as np

def form_patches(ds, patch_sizes=(65, 128)):
    
    img_patches, seg_patches = [], []
    patch_reshape_size = 64

    for (img, seg) in ds:
        img, seg = img.numpy(), seg.numpy()
        
        img_width, img_height = tf.shape(img)[:2]
        
        nsamples = 256 
        opts = list(range(patch_sizes[0], patch_sizes[1]+1, 2))
        sizes = np.random.choice(opts,nsamples, p=[1/len(opts)]*len(opts))
        for size in sizes:
            size = size//2 #np.random.randint(patch_sizes[0], patch_sizes[1]) // 2
            x = np.random.randint(size, img_width - size)
            y = np.random.randint(size, img_height - size)
            
            patch_img = tf.cast(tf.image.resize(img[x - size:x + size, y - size:y + size], (patch_reshape_size, patch_reshape_size), 'nearest'), img.dtype)
            patch_seg = tf.cast(tf.image.resize(seg[x - size:x + size, y - size:y + size], (patch_reshape_size, patch_reshape_size), 'nearest'), seg.dtype)
            
            img_patches.append(patch_img)
            seg_patches.append(patch_seg)
        
    # Shuffle patches
    indices = list(range(len(img_patches)))
    np.random.shuffle(indices)
    img_patches = np.array(img_patches)[indices]
    seg_patches = np.array(seg_patches)[indices]
    
    # Stitch patches
    stitched_imgs = []
    stitched_segs = []
    img_width, img_height = 512, 512
    npatches = (img_width // patch_reshape_size) * (img_height // patch_reshape_size)  # Number of patches
    for i in range(0, len(indices)-npatches, npatches):
        img_stitched = np.zeros((img_width, img_height, 1), dtype=img.dtype)
        seg_stitched = np.zeros((img_width, img_height, 1), dtype=seg.dtype)
        
        for j, patch_idx in enumerate(range(i, i + npatches)):
            x = (j // (img_height // patch_reshape_size)) * patch_reshape_size
            y = (j % (img_height // patch_reshape_size)) * patch_reshape_size
            img_stitched[x:x + patch_reshape_size, y:y + patch_reshape_size, 0] = img_patches[patch_idx][:, :, 0]
            seg_stitched[x:x + patch_reshape_size, y:y + patch_reshape_size, 0] = seg_patches[patch_idx][:, :, 0]
        
        stitched_imgs.append(img_stitched)
        stitched_segs.append(seg_stitched)
    
    stitched_imgs = np.array(stitched_imgs)
    stitched_segs = np.array(stitched_segs) #thresholding
    
    return tf.data.Dataset.from_tensor_slices((stitched_imgs, stitched_segs))

