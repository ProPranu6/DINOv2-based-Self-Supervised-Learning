from tensorflow.keras import backend as K
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

def dice_loss(y_true, y_pred):
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(K.flatten(y_pred), tf.float32)
    intersection = K.sum(y_true_f*y_pred_f)

    val = (2. * intersection + K.epsilon()) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + K.epsilon())
    return 1. - val

def generalized_dice_loss(y_true, y_pred):
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(K.flatten(y_pred), tf.float32)

    # Compute the weights
    w = 1.0 / (K.sum(y_true_f) ** 2 + K.epsilon())
    

    intersection = K.sum(w * y_true_f * y_pred_f)
    union = K.sum(w * y_true_f + w * y_pred_f)
    
    gdl = 1.0 - (2.0 * intersection + K.epsilon()) / (union + K.epsilon())
    
    return gdl



def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss for binary classification.
    
    Args:
    gamma (float): Focusing parameter. Default is 2.0.
    alpha (float): Balance parameter. Default is 0.25.
    
    Returns:
    loss function: A callable loss function.
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        Focal Loss function.
        
        Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
        Returns:
        loss: Computed focal loss value.
        """
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

        return K.mean(fl)
    
    return focal_loss_fixed


def combined_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Combined Focal and Dice Loss function.
    
    Args:
    y_true: Ground truth labels.
    y_pred: Predicted labels.
    gamma (float): Focusing parameter for focal loss. Default is 2.0.
    alpha (float): Balance parameter for focal loss. Default is 0.25.
    smooth (float): Smoothing factor for dice loss. Default is 1e-6.
    
    Returns:
    loss: Combined focal and dice loss value.
    """
    fl = focal_loss(gamma=gamma, alpha=alpha)(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return fl + dl

# Example usage in a Keras model:
# model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])
