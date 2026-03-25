# CW LAB

[DEMO](https://e04.github.io/web-deep-cw-decoder/)

<img width="825" height="514" alt="web-deep-cw" src="https://github.com/user-attachments/assets/a224be0a-a685-4dd8-be99-d0f376a43aa2" />

This is a web-based, real-time Morse code (CW) decoder powered by a CRNN (Convolutional Recurrent Neural Network) model with a CTC Loss function.

A key feature of this application is its client-side processing architecture. By leveraging ONNX Runtime Web, the entire decoding process runs completely within your browser.

The neural network model has been trained on an extensive dataset of 50 hours of programmatically generated Morse code audio, enabling it to achieve high accuracy across various sending speeds and conditions.

## Features

- **Real-time Morse code decoding** using machine learning
- **Audio visualization** with spectrum scope style display
- **Browser-based** - no installation required
- **Multiplatform** - supports Windows/mac/Android/iOS devices
- **Signal quality monitoring** - SNR and confidence indicators
- **Synthetic data generation** - built-in tools for training data augmentation
- **CTC decoding** - proper Connectionist Temporal Classification decoding
- **Dual decoder mode** - switch between deep learning and traditional DSP approaches

## Usage

Open this page:

[https://e04.github.io/web-deep-cw-decoder/](https://e04.github.io/web-deep-cw-decoder/)

## Acknowledgments

### ggmorse

This project incorporates ideas and techniques from [ggmorse](https://github.com/ggerganov/ggmorse) by [Georgi Gerganov](https://github.com/ggerganov), a high-performance Morse code decoding library written in C++.

Key inspirations from ggmorse:
- **Goertzel algorithm** for efficient single-tone detection
- **Automatic pitch and speed detection** approaches
- **Signal processing pipeline** design patterns

ggmorse is licensed under the MIT License.

### DeepCW

This project draws inspiration from [DeepCW](https://github.com/VE3NEA/DeepCW) by Alex Shovkoplyas (VE3NEA), a deep learning-based Morse code decoder using TensorFlow.

Key learnings from DeepCW:
- **Spectrogram-based input representation**
- **Synthetic data generation** with realistic noise, WPM variation, and keying styles
- **Character frequency distributions** based on real-world ham radio statistics
- **CTC Loss** for sequence-to-sequence learning

### Training Data Generation

The synthetic data generation pipeline was inspired by techniques from [Morse Code Deep Learning Detect and Decode](https://github.com/MaorAssayag/morse-deep-learning-detect-and-decode) by Maor Assayag et al., developed for the Technion's EE Deep Learning course.

## License

This project is licensed under the MIT License.
