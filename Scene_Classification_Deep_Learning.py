import os
import argparse
import tensorflow as tf
from vgg_model import VGGModel
from SceneClassificationModel import SceneClassificationModel
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger
import datetime

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train or evaluate neural network models.")
    
    parser.add_argument('--task', required=True, choices=['1', '2'], 
                        help="Specify the task: 1 (train from scratch) or 2 (fine-tune VGG-16).")
    parser.add_argument('--data', default=os.path.join(os.getcwd(), '../data/'), 
                        help="Path to the dataset.")
    parser.add_argument('--load-vgg', default=os.path.join(os.getcwd(), 'vgg16_imagenet.h5'), 
                        help="Path to pre-trained VGG-16 weights (Task 2 only).")
    parser.add_argument('--load-checkpoint', default=None, 
                        help="Path to model checkpoint (.h5) to resume training or evaluate.")
    parser.add_argument('--confusion', action='store_true', 
                        help="Log a confusion matrix at the end of each epoch (TensorBoard).")
    parser.add_argument('--evaluate', action='store_true', 
                        help="Evaluate the model without training (requires a checkpoint).")
    
    return parser.parse_args()

def train(model, datasets, checkpoint_path):
    """Train the model with the specified datasets and save checkpoints."""
    # Define Keras callbacks
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                checkpoint_path, "weights.e{epoch:02d}-acc{val_sparse_categorical_accuracy:.4f}.h5"
            ),
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch', profile_batch=0),
        ImageLabelingLogger(datasets, log_dir=log_dir)
    ]

    # Add confusion matrix logger if enabled
    if ARGS.confusion:
        callbacks.append(ConfusionMatrixLogger(datasets))

    # Train the model
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callbacks
    )

def test(model, test_data):
    """Evaluate the model on the test dataset."""
    model.evaluate(x=test_data, verbose=1)

def main():
    """Main execution function."""
    tf.compat.v1.enable_eager_execution()  # Enable TensorFlow eager execution
    tf.config.run_functions_eagerly(False)  # Ensure functions run eagerly

    # Load datasets based on task
    datasets = Datasets(ARGS.data, ARGS.task)

    # Initialize model and checkpoint paths
    if ARGS.task == '1':
        model = SceneClassificationModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "./sc_model_checkpoints/"
        model.summary()
    else:
        model = VGGModel()
        model(tf.keras.Input(shape=(224, 224, 3)))
        checkpoint_path = "./vgg_model_checkpoints/"
        model.summary()
        if ARGS.load_checkpoint is None:
            model.load_weights(ARGS.load_vgg, by_name=True)

    # Load model checkpoint if specified
    if ARGS.load_checkpoint:
        print(ARGS.load_checkpoint)
        model.load_weights(ARGS.load_checkpoint)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_path, exist_ok=True)

    # Compile the model
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"]
    )

    # Train or evaluate the model
    if ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path)

# main function
if __name__ == '__main__':
    ARGS = parse_args()
    main()
