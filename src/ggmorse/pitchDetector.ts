/**
 * Pitch Detector for CW/Morse Code
 *
 * Uses STFT (Short-Time Fourier Transform) with peak detection
 */

import { MultiFrequencyGoertzel } from "./goertzel";

export interface PitchDetectionConfig {
  sampleRate: number;
  fftSize: number;
  hopSize: number;
  minFrequency: number;
  maxFrequency: number;
  noiseThreshold: number;
}

export interface PitchResult {
  frequency: number | null;
  magnitude: number;
  confidence: number;
  isActive: boolean;
}

function computeMagnitudeSpectrum(
  real: Float32Array,
  imag: Float32Array
): Float32Array {
  const N = real.length;
  const magnitude = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }
  return magnitude;
}

function applyHammingWindow(samples: Float32Array): Float32Array {
  const N = samples.length;
  const windowed = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const multiplier = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (N - 1));
    windowed[i] = samples[i] * multiplier;
  }
  return windowed;
}

function fft(real: Float32Array, imag: Float32Array): void {
  const N = real.length;
  const n = Math.log2(N);

  for (let i = 0; i < N; i++) {
    const j = reverseBits(i, n);
    if (j > i) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let size = 2; size <= N; size *= 2) {
    const halfSize = size / 2;
    const angleStep = (2 * Math.PI) / size;

    for (let i = 0; i < N; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const angle = angleStep * j;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        const re = cos * real[i + j + halfSize] - sin * imag[i + j + halfSize];
        const im = sin * real[i + j + halfSize] + cos * imag[i + j + halfSize];

        real[i + j + halfSize] = real[i + j] - re;
        imag[i + j + halfSize] = imag[i + j] - im;
        real[i + j] = real[i + j] + re;
        imag[i + j] = imag[i + j] + im;
      }
    }
  }
}

function reverseBits(x: number, bits: number): number {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

export class PitchDetector {
  private config: PitchDetectionConfig;
  private history: Float32Array;
  private historyIndex: number = 0;
  private spectrogram: Float32Array[] = [];
  private lastPitch: PitchResult;

  constructor(config: PitchDetectionConfig) {
    this.config = config;
    this.history = new Float32Array(config.fftSize);
    this.lastPitch = {
      frequency: null,
      magnitude: 0,
      confidence: 0,
      isActive: false,
    };
  }

  processSamples(samples: Float32Array): PitchResult[] {
    const results: PitchResult[] = [];
    const { fftSize, hopSize } = this.config;

    for (let i = 0; i < samples.length; i += hopSize) {
      for (let j = 0; j < fftSize; j++) {
        const idx = (this.historyIndex + j) % fftSize;
        if (i + j < samples.length) {
          this.history[idx] = samples[i + j];
        }
      }
      this.historyIndex = (this.historyIndex + hopSize) % fftSize;

      const window = new Float32Array(fftSize);
      for (let j = 0; j < fftSize; j++) {
        window[j] = this.history[(this.historyIndex + j) % fftSize];
      }

      const windowed = applyHammingWindow(window);
      const real = windowed;
      const imag = new Float32Array(fftSize);
      fft(real, imag);

      const magnitude = computeMagnitudeSpectrum(real, imag);
      const pitchResult = this.findPeak(magnitude);
      results.push(pitchResult);
      this.lastPitch = pitchResult;

      this.spectrogram.push(magnitude.slice(0, fftSize / 2));
      if (this.spectrogram.length > 100) {
        this.spectrogram.shift();
      }
    }

    return results;
  }

  private findPeak(magnitude: Float32Array): PitchResult {
    const { sampleRate, fftSize, minFrequency, maxFrequency, noiseThreshold } = this.config;

    const binSize = sampleRate / fftSize;
    const minBin = Math.floor(minFrequency / binSize);
    const maxBin = Math.ceil(maxFrequency / binSize);

    let maxMag = noiseThreshold;
    let maxBinIndex = -1;

    for (let i = minBin; i <= maxBin && i < magnitude.length; i++) {
      if (magnitude[i] > maxMag) {
        maxMag = magnitude[i];
        maxBinIndex = i;
      }
    }

    let frequency: number | null = null;
    let confidence = 0;

    if (maxBinIndex > 0 && maxBinIndex < magnitude.length - 1) {
      const alpha = magnitude[maxBinIndex - 1];
      const beta = magnitude[maxBinIndex];
      const gamma = magnitude[maxBinIndex + 1];

      const p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);
      frequency = (maxBinIndex + p) * binSize;

      const peakToSide = Math.max(
        Math.abs(beta - alpha),
        Math.abs(beta - gamma)
      );
      confidence = Math.min(1, peakToSide / (beta + 1e-10));
    } else if (maxBinIndex >= 0) {
      frequency = maxBinIndex * binSize;
      confidence = 0.5;
    }

    return {
      frequency,
      magnitude: maxMag,
      confidence,
      isActive: maxBinIndex >= 0,
    };
  }

  getStablePitch(samples: Float32Array): PitchResult {
    const results = this.processSamples(samples);
    if (results.length === 0) {
      return this.lastPitch;
    }

    let sumFreq = 0;
    let sumMag = 0;
    let sumConf = 0;
    let activeCount = 0;

    for (const r of results) {
      if (r.isActive) {
        sumFreq += r.frequency!;
        sumMag += r.magnitude;
        sumConf += r.confidence;
        activeCount++;
      }
    }

    if (activeCount === 0) {
      return {
        frequency: null,
        magnitude: 0,
        confidence: 0,
        isActive: false,
      };
    }

    return {
      frequency: sumFreq / activeCount,
      magnitude: sumMag / activeCount,
      confidence: sumConf / activeCount,
      isActive: true,
    };
  }

  getSpectrogram(): Float32Array[] {
    return this.spectrogram;
  }

  reset(): void {
    this.history.fill(0);
    this.historyIndex = 0;
    this.spectrogram = [];
    this.lastPitch = {
      frequency: null,
      magnitude: 0,
      confidence: 0,
      isActive: false,
    };
  }
}

export class GoertzelPitchDetector {
  private config: PitchDetectionConfig;
  private goertzel: MultiFrequencyGoertzel;

  constructor(config: PitchDetectionConfig) {
    this.config = config;

    const binSize = config.sampleRate / config.fftSize;
    const frequencies: number[] = [];
    for (
      let f = config.minFrequency;
      f <= config.maxFrequency;
      f += binSize
    ) {
      frequencies.push(Math.round(f));
    }

    this.goertzel = new MultiFrequencyGoertzel(
      config.sampleRate,
      config.fftSize,
      frequencies
    );
  }

  detect(samples: Float32Array): PitchResult {
    this.goertzel.reset();
    this.goertzel.processSamples(samples);

    const result = this.goertzel.getMaxFrequency(this.config.noiseThreshold);

    if (!result) {
      return {
        frequency: null,
        magnitude: 0,
        confidence: 0,
        isActive: false,
      };
    }

    return {
      frequency: result.frequency,
      magnitude: result.magnitude,
      confidence: Math.min(1, result.magnitude / 10000),
      isActive: true,
    };
  }
}
