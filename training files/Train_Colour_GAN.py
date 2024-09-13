import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
import keras
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from scipy.stats import entropy
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, image_size=(128, 128), test_size=0.2, random_state=None):
    images = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img)
            images.append(img_array)
    images = np.asarray(images)

    # Split the dataset into train, test, and validation sets
    X_train, X_test = train_test_split(images, test_size=test_size, random_state=random_state)

    return X_train, X_test

folder_path = "More64/"
X_train, X_test = load_images_from_folder(folder_path, test_size=0.1)
X_train = (X_train / 127.5) - 1.0
X_test = (X_test / 127.5) - 1.0
latent_dim = 100

def save_images(epoch, images):
    """Saves generated images to a specified path.

    Args:
        epoch (int): The current training epoch.
        images (np.ndarray): The generated images to save.
        path (str, optional): The path to save the images. Defaults to "./images/".
    """

    # change to where ever you want to save training predictions
    predictions_path = "training_predictions"

    os.makedirs(predictions_path, exist_ok=True)
    os.makedirs(predictions_path + str(epoch) + "/", exist_ok=True)
    for i in range(len(images)):
        save_img(os.path.join(predictions_path + str(epoch), f"{str(i)}_saved.png"), images[i])

def make_discriminator():
    """Creates a convolutional neural network for image classification.

    Returns:
        tf.keras.Sequential: The compiled discriminator model.
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, kernel_size=4, strides=2, padding="same", input_shape=(128, 128, 1)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def make_generator():
    """Creates a convolutional neural network for image generation.

    Args:
        latent_dim (int, optional): The dimensionality of the latent space. Defaults to 100.

    Returns:
        tf.keras.Sequential: The compiled generator model.
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 128, use_bias=False, input_shape=(100,)))
    model.add(layers.Reshape((8, 8, 128)))
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(256, kernel_size=1, strides=1, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(256, kernel_size=1, strides=1, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))        
    model.add(layers.Conv2DTranspose(512, kernel_size=1, strides=1, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"))
    return model

class GAN(keras.Model):
    """Generative Adversarial Network (GAN) class.

    This class combines a discriminator and a generator model for image generation.

    Args:
        discriminator (tf.keras.Sequential): The discriminator model.
        generator (tf.keras.Sequential): The generator model.
        latent_dim (int, optional): The dimensionality of the latent space. Defaults to 100.
    """

    def __init__(self, discriminator, generator, latent_dim=100):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = np.random.seed(1234)
        self.loss_fn = keras.losses.BinaryCrossentropy()
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        """Compiles the GAN model with specific optimizers and loss function.

        Args:
            loss_fn (callable, optional): The loss function for both discriminator and generator. Defaults to keras.losses.BinaryCrossentropy().
            d_optimizer (str, optional): The optimizer for the discriminator. Defaults to "adam".
            g_optimizer (str, optional): The optimizer for the generator. Defaults to "adam".
        """
        super().compile()
        self.loss_fn = loss_fn
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        """Returns the list of metrics used during training."""
        return [self.d_loss_metric, self.g_loss_metric]

    def drop_lr(self, rate=0.00001):
        self.d_optimizer.learning_rate.assign(rate)
        self.g_optimizer.learning_rate.assign(rate)

    def generate_noise(self, batch_size):
        """Samples random points in the latent space.

        Args:
            batch_size (int): The batch size for the training step.

        Returns:
            tf.Tensor: A tensor of random latent vectors.
        """
        return tf.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)

    def generate_fake_images(self, noise):
        """Decodes latent vectors to fake images.

        Args:
            noise (tf.Tensor): A tensor of random latent vectors.

        Returns:
            tf.Tensor: A tensor of generated images.
        """
        return self.generator(noise)

    def create_combined_images(self, real_images, fake_images):
        """Combines real and fake images for training.

        Args:
            real_images (tf.Tensor): A tensor of real images.
            fake_images (tf.Tensor): A tensor of generated images.

        Returns:
            tf.Tensor: A tensor of combined real and fake images.
        """
        return keras.layers.concatenate([fake_images, real_images], axis=0)

    def create_discriminate_labels(self, batch_size):
        """Assembles labels discriminating real from fake images.

        Args:
            batch_size (int): The batch size for the training step.

        Returns:
            tf.Tensor: A tensor of labels for real and fake images.
        """
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        return keras.layers.concatenate([real_labels, fake_labels], axis=0)

    def add_noise_to_labels(self, labels):
        """Adds random noise to the labels for improved training.

        Args:
            labels (tf.Tensor): A tensor of labels.

        Returns:
            tf.Tensor: A tensor of labels with added noise.
        """
        return labels + 0.05 * tf.random.uniform(tf.shape(labels))

    def train_discriminator(self, combined_images, labels):
        """Trains the discriminator to distinguish real from fake images.

        Args:
            combined_images (tf.Tensor): A tensor of combined real and fake images.
            labels (tf.Tensor): A tensor of labels for real and fake images.

        Returns:
            float: The discriminator loss.
        """
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        return d_loss


    def train_generator(self, noise, batch_size):
        """Trains the generator to fool the discriminator.

        Args:
            noise (tf.Tensor): A tensor of random latent vectors.

        Returns:
            float: The generator loss.
        """
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(noise))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss

    def train_step(self, real_images):
        """Performs a single training step for the GAN.

        Args:
            real_images (tf.Tensor): A batch of real images.

        Returns:
            dict: A dictionary containing the discriminator and generator losses.
        """
        batch_size = tf.shape(real_images)[0]
        noise = self.generate_noise(batch_size)
        fake_images = self.generate_fake_images(noise)
        combined_images = self.create_combined_images(real_images, fake_images)
        labels = self.create_discriminate_labels(batch_size)
        labels = self.add_noise_to_labels(labels)

        d_loss = self.train_discriminator(combined_images, labels)
        g_loss = self.train_generator(noise, batch_size)

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

    def create_ims(self, num_images=8):
        inputs = tf.random.normal(shape=(num_images, self.latent_dim))
        images = self.generator(inputs)
        return images


# set correct path
model_path = "/content/drive/MyDrive/2024_latent_diffusion_model1.0/train_pred_256_3x3/BNW_models/"

def plot_images(images, num_rows=2, num_cols=4, figsize=(10, 5)):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')  # Assuming images are grayscale
        ax.axis('off')
    plt.tight_layout()
    plt.show()

class Monitor(keras.callbacks.Callback):
    def __init__(self, save_models, show_images):
        self.save_models = save_models
        self.show_images = show_images
        self.seed_generator = np.random.seed(1234)

    def on_epoch_end(self, epoch, show_images=False, save_model=False, logs=None):
        if save_model:
            self.model.generator.save(model_path + "gen_model_" + str(epoch) + ".h5")
            self.model.discriminator.save(model_path + "dis_model_" + str(epoch) + ".h5")
        if show_images:
            random_latent_vectors = tf.random.normal(
                # 8 images, with input noise dim of 100
                shape=(8, 100), seed=self.seed_generator
            )
            generated_images = (self.model.generator(random_latent_vectors)*255).numpy()
            plot_images(generated_images)

learning_rate = 1e-4
generator = make_generator()
discriminator = make_discriminator()
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=100)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    X_train, epochs=50, callbacks=[Monitor(save_models=False, show_images=False)]
)

# lower learning rate to 0.00001 after 100 epochs for more fine grained training
gan.drop_lr()

gan.fit(
    X_train, epochs=50, callbacks=[Monitor(save_models=False, show_images=False)]
)

# the following code is used to test the latent diffusion model too but there is no use
# repeating it, it is left the the BnWGAN code as that is subtly different
def scale_pixels(images): 
    return ((images + 1) * 127.5)  # Scale images from [-1, 1] to [0, 255]

def scale_size(images):
    images = tf.image.resize(images, (299, 299))
    return preprocess_input(images)


def compute_is(images, num_splits=10, batch_size=4):
    """Compute the Inception Score."""
    # Load InceptionV3 model
    inception_model = InceptionV3(include_top=False, pooling='avg')

    # Split images into smaller batches
    image_batches = tf.split(images, num_or_size_splits=num_splits)

    scores = []
    for batch in image_batches:
        # Preprocess batch
        batch = scale_pixels(batch)
        batch = scale_size(batch)
        # Compute features
        features = inception_model.predict(batch)
        # Compute probabilities
        probs = np.exp(features) / np.sum(np.exp(features), axis=1, keepdims=True)
        # Compute scores
        scores.append(probs)
    
    # Combine scores from all batches
    preds = np.concatenate(scores, axis=0)

    # Compute marginal entropy and conditional entropy
    p_yx = preds
    p_y = np.mean(p_yx, axis=0)
    marginal_entropy = entropy(p_y)
    conditional_entropy = np.mean(entropy(p_yx.T))

    # Compute Inception Score
    is_score = np.exp(conditional_entropy - marginal_entropy)
    return is_score

def compute_fid(real_images, generated_images, batch_size=4):
    """Compute the Fréchet Inception Distance."""
    # Load InceptionV3 model
    inception_model = InceptionV3(include_top=False, pooling='avg')

    # Split images into smaller batches
    real_batches = tf.split(real_images, num_or_size_splits=(len(real_images) + batch_size - 1) // batch_size)
    gen_batches = tf.split(generated_images, num_or_size_splits=(len(generated_images) + batch_size - 1) // batch_size)

    real_features = []
    for batch in real_batches:
        # Preprocess batch
        batch = scale_size(batch)
        # Compute features
        features = inception_model.predict(batch)
        real_features.append(features)
    real_features = np.concatenate(real_features, axis=0)

    gen_features = []
    for batch in gen_batches:
        # Preprocess batch
        # batch = preprocess_images(batch)
        batch = scale_pixels(batch)
        batch = scale_size(batch)
        # Compute features
        features = inception_model.predict(batch)
        gen_features.append(features)
    gen_features = np.concatenate(gen_features, axis=0)

    # Compute mean and covariance of real and generated features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    # Compute squared Frobenius norm between means and matrix trace of covariance product
    fid_score = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * sqrtm(sigma_real.dot(sigma_gen)))
    return fid_score

#     return fid_score
def generate_images(generator, num_images, batch_size=32):
    """Generate images using the generator model."""
    generated_images = []
    for _ in range(num_images // batch_size):
        noise = tf.random.normal(shape=(batch_size, 100))
        batch_generated_images = generator(noise)
        generated_images.append(batch_generated_images)
    if num_images % batch_size != 0:
        noise = tf.random.normal(shape=((num_images % batch_size), 100))
        batch_generated_images = generator(noise)
        generated_images.append(batch_generated_images)
    return tf.concat(generated_images, axis=0)
# Example usage
# Assuming you have a generator model named generator
X_test = X_test[0:1900]
num_images = len(X_test)  # Adjust as needed
print(num_images)
# Load real images for FID calculation
images = generate_images(gan.generator, num_images)
# Compute scores
is_score = compute_is(images)
fid_score = compute_fid(X_test, images)
print(is_score)
print(fid_score)

# show models final predictions
images = gan.create_ims()
plot_images(images)