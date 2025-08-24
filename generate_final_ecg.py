# generate_final_ecg.py
# This script loads the trained generator model and creates
# a plot of newly generated synthetic ECG heartbeats.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_PATH = os.path.join('training_output', 'generator_final.keras') # Or another saved model
NOISE_DIM = 100
NUM_SAMPLES = 10 # Number of ECGs to generate

def generate_ecgs():
    """
    Loads the generator model and generates new ECG samples.
    """
    print(f"--- Loading Model from {MODEL_PATH} ---")
    try:
        # It's good practice to specify custom objects if the model uses them,
        # but for this simple model, it might not be necessary.
        # If you face an error here, you might need to load it without compiling.
        generator = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have a trained model saved at the specified path.")
        return

    # Generate random noise as input for the generator
    random_noise = tf.random.normal([NUM_SAMPLES, NOISE_DIM])

    print("\n--- Generating New ECG Samples ---")
    # Get the model's predictions
    generated_ecgs = generator.predict(random_noise)

    print("\n--- Plotting and Saving Results ---")
    # Create a plot to display the results
    plt.figure(figsize=(15, 8))
    plt.suptitle('Final Generated Synthetic ECG Heartbeats', fontsize=16)

    for i in range(NUM_SAMPLES):
        plt.subplot(2, 5, i + 1)
        plt.plot(generated_ecgs[i, :, 0])
        plt.title(f'Sample {i+1}')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the final plot
    output_filename = 'final_generated_ecgs.png'
    plt.savefig(output_filename)
    print(f"Final plot saved as '{output_filename}'")
    
    # Show the plot on screen
    plt.show()


if __name__ == '__main__':
    generate_ecgs()

