"""
Convert a trained .keras model into TensorFlow.js format so it can run
directly in the browser (used by frontend/script.js).

Usage:
    pip install tensorflowjs
    python convert_to_tfjs.py --input drowsiness_model.keras --output ../frontend/model
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the .keras model file")
    parser.add_argument("--output", required=True, help="Output directory for TF.js model")
    args = parser.parse_args()

    import tensorflowjs as tfjs
    import tensorflow as tf

    print(f"Loading model from {args.input} ...")
    model = tf.keras.models.load_model(args.input)

    os.makedirs(args.output, exist_ok=True)
    print(f"Converting and saving TF.js model to {args.output} ...")
    tfjs.converters.save_keras_model(model, args.output)

    print("Done. Files created:")
    for f in os.listdir(args.output):
        print(f"  - {f}")
    print("\nCopy/keep this folder at frontend/model/ before deploying.")


if __name__ == "__main__":
    main()
