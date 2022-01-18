import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') #(60000, 28, 28, 1) to (60000, 28, 28) (channel 제거)
train_images = (train_images - 127.5) / 127.5  #0~255 to -1~1

BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 1000
noise_dim = 100

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = tf.keras.Sequential([
    layers.Input(100, BATCH_SIZE),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(784, activation='tanh')
    ])

discriminator = tf.keras.Sequential([
    layers.Input(784, BATCH_SIZE),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(1)
    ])

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def save_fig(epoch):
    size = 10
    pred = generator( tf.random.normal([size ** 2, noise_dim])).numpy().reshape(-1, 28, 28)

    fig = plt.figure(figsize = (size, size))
    for i in range(size ** 2):
        plt.subplot(size, size, i + 1)
        plt.imshow(pred[i], cmap = 'gray')
        plt.axis('off')
    plt.savefig('mnist_gan_epoch={}.png'.format(epoch))


for epoch in tqdm(range(EPOCHS)):
    for image_batch in train_dataset:
        image_batch = image_batch.numpy().reshape(-1, 784)
        train_step(image_batch)
    if epoch>10 and epoch%25 == 0:
        save_fig(epoch)

