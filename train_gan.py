# train_gan.py
# This script builds and trains the Generative Adversarial Network (GAN)
# to generate synthetic ECG heartbeats.
# VERSION 2.0: Now includes checkpointing to save and resume training.

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time # Import time for tracking epoch duration

# --- Configuration ---
PROCESSED_DATA_FILE = 'processed_ecg_data.npy'
OUTPUT_DIR = 'training_output'
CHECKPOINT_DIR = 'training_checkpoints' # Directory to save checkpoints
IMAGE_SAVE_INTERVAL = 50  # Save a plot of generated ECGs every 50 epochs
CHECKPOINT_SAVE_INTERVAL = 10 # Save a checkpoint every 10 epochs
EPOCHS = 5000
BATCH_SIZE = 128
NOISE_DIM = 100  # The dimension of the random noise input to the generator

# --- Create Output Directories ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# --- 1. Build the Generator ---
# The generator takes random noise and tries to create a realistic ECG signal.
def build_generator(input_shape):
    model = tf.keras.Sequential(name='Generator')
    model.add(layers.Dense(45 * 256, use_bias=False, input_shape=(input_shape,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((45, 256)))
    model.add(layers.Conv1DTranspose(128, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv1DTranspose(64, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv1DTranspose(1, 5, strides=1, padding='same', use_bias=False, activation='tanh'))
    return model

# --- 2. Build the Discriminator ---
# The discriminator takes an ECG signal (real or fake) and tries to classify it.
def build_discriminator(input_shape):
    model = tf.keras.Sequential(name='Discriminator')
    model.add(layers.Conv1D(64, 5, strides=2, padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# --- 3. Define Loss and Optimizers ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- 4. The Training Step ---
@tf.function
def train_step(ecg_signals, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_ecgs = generator(noise, training=True)
        real_output = discriminator(ecg_signals, training=True)
        fake_output = discriminator(generated_ecgs, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# --- 5. Function to Save Generated Plots ---
def save_generated_plots(generator, epoch, test_noise):
    generated_ecgs = generator(test_noise, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.plot(generated_ecgs[i, :, 0])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(os.path.join(OUTPUT_DIR, f'ecg_at_epoch_{epoch:04d}.png'))
    plt.close()

# --- 6. The Main Training Function ---
def train():
    print("\n--- Loading Data ---")
    train_data = np.load(PROCESSED_DATA_FILE)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data)).batch(BATCH_SIZE)
    ecg_shape = (train_data.shape[1], train_data.shape[2])

    print("\n--- Building Models ---")
    generator = build_generator(NOISE_DIM)
    discriminator = build_discriminator(ecg_shape)
    
    # --- Set up Checkpointing ---
    # Create a checkpoint object that tracks the models and their optimizers
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    # Create a manager to handle saving and restoring checkpoints
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)

    # --- Restore from the latest checkpoint, if one exists ---
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
        # Extract the last epoch number from the checkpoint filename
        start_epoch = int(checkpoint_manager.latest_checkpoint.split('-')[-1])
    else:
        print("Initializing from scratch.")
        start_epoch = 0

    seed = tf.random.normal([16, NOISE_DIM])

    print(f"\n--- Starting Training from Epoch {start_epoch + 1} ---")
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        gen_loss_epoch = []
        disc_loss_epoch = []
        
        for ecg_batch in train_dataset:
            gen_loss, disc_loss = train_step(ecg_batch, generator, discriminator)
            gen_loss_epoch.append(gen_loss)
            disc_loss_epoch.append(disc_loss)

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} - Gen Loss: {np.mean(gen_loss_epoch):.4f}, Disc Loss: {np.mean(disc_loss_epoch):.4f} ({epoch_duration:.2f} sec)")

        # Save generated image samples periodically
        if (epoch + 1) % IMAGE_SAVE_INTERVAL == 0:
            save_generated_plots(generator, epoch + 1, seed)
        
        # Save a checkpoint periodically
        if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            checkpoint_manager.save(checkpoint_number=epoch + 1)
            print(f"Saved checkpoint for epoch {epoch + 1}")

    # Save the final generator model for easy use later
    generator.save(os.path.join(OUTPUT_DIR, 'generator_final.keras'))
    print("\n--- Training Finished ---")


if __name__ == '__main__':
    train()
