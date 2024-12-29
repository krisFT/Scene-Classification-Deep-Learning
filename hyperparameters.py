# Image size for task 1
img_size = 224  # Task 2 image size is hardcoded to 224

# Number of image scene classes (do not change)
category_num = 15

# Sample size for calculating mean and standard deviation of training data
preprocess_sample_size = 400

# Training parameters
num_epochs = 20  # Increase if using more complex networks or regularization
batch_size = 10  # Number of training examples per batch
learning_rate = 1e-4  # Critical for training success
momentum = 0.09  # For momentum-based optimizers
