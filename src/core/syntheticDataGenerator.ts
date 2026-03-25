/**
 * Synthetic Morse Code Data Generator
 *
 * Generates large-scale synthetic training data for the CW decoder model.
 * Produces audio-text pairs with configurable WPM, tone frequency, and SNR.
 */

import { MorseEncoder } from "./morseEncoder";

export interface SyntheticConfig {
  /** Words per minute (8-50 range) */
  wpm: number;
  /** Tone frequency in Hz (400-1200 range) */
  toneHz: number;
  /** Signal-to-noise ratio in dB (-10 to 40) */
  snrDb: number;
  /** Sample rate for output audio (must be 3200 for model compatibility) */
  sampleRate: number;
  /** Farnsworth spacing multiplier */
  farnsworth: number;
}

export interface SyntheticResult {
  /** Audio samples at target sampleRate (3200 Hz) */
  audio: Float32Array;
  /** Ground truth text label */
  label: string;
  /** Generation config used */
  config: SyntheticConfig;
  /** Duration in seconds */
  duration: number;
}

/** Default configuration for synthetic data generation */
export const SYNTHETIC_DEFAULTS: SyntheticConfig = {
  wpm: 20,
  toneHz: 700,
  snrDb: 20,
  sampleRate: 3200,
  farnsworth: 1.0,
};

export class SyntheticDataGenerator {
  private morseEncoder: MorseEncoder;
  private config: SyntheticConfig;

  constructor(config: Partial<SyntheticConfig> = {}) {
    this.config = { ...SYNTHETIC_DEFAULTS, ...config };
    this.morseEncoder = new MorseEncoder({
      wpm: this.config.wpm,
      toneHz: this.config.toneHz,
      sampleRate: 48000, // Generate at 48kHz, then resample
      farnsworth: this.config.farnsworth,
    });
  }

  /**
   * Update configuration
   */
  setConfig(config: Partial<SyntheticConfig>): void {
    this.config = { ...this.config, ...config };
    this.morseEncoder.setConfig({
      wpm: this.config.wpm,
      toneHz: this.config.toneHz,
      farnsworth: this.config.farnsworth,
    });
  }

  /**
   * Get current configuration
   */
  getConfig(): SyntheticConfig {
    return { ...this.config };
  }

  /**
   * Resample audio from 48kHz to target sample rate using linear interpolation
   */
  private resample(audio: Float32Array, fromRate: number, toRate: number): Float32Array {
    if (fromRate === toRate) return audio;

    const ratio = fromRate / toRate;
    const outputLen = Math.floor(audio.length / ratio);
    const output = new Float32Array(outputLen);

    for (let i = 0; i < outputLen; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, audio.length - 1);
      const frac = srcIndex - srcIndexFloor;
      output[i] = audio[srcIndexFloor] * (1 - frac) + audio[srcIndexCeil] * frac;
    }

    return output;
  }

  /**
   * Generate Gaussian white noise scaled to achieve target SNR
   */
  private addNoise(signal: Float32Array, snrDb: number): Float32Array {
    // Calculate signal power
    let signalPower = 0;
    for (let i = 0; i < signal.length; i++) {
      signalPower += signal[i] * signal[i];
    }
    signalPower /= signal.length;

    // Calculate noise standard deviation from SNR
    // SNR_dB = 10 * log10(signal_power / noise_power)
    // noise_power = signal_power / 10^(SNR_dB / 10)
    // noise_std = sqrt(noise_power)
    const noisePower = signalPower / Math.pow(10, snrDb / 10);
    const noiseStd = Math.sqrt(noisePower);

    // Generate Gaussian noise using Box-Muller transform
    const noise = new Float32Array(signal.length);
    for (let i = 0; i < signal.length; i += 2) {
      // Generate two independent Gaussian samples
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

      noise[i] = noiseStd * z0;
      if (i + 1 < signal.length) {
        noise[i + 1] = noiseStd * z1;
      }
    }

    // Add noise to signal
    const output = new Float32Array(signal.length);
    for (let i = 0; i < signal.length; i++) {
      output[i] = signal[i] + noise[i];
    }

    return output;
  }

  /**
   * Generate synthetic Morse code audio with configurable parameters
   */
  generate(text: string, config?: Partial<SyntheticConfig>): SyntheticResult {
    // Merge config
    const mergedConfig = { ...this.config, ...config };
    this.setConfig(mergedConfig);

    // Generate clean audio at 48kHz
    const cleanAudio48k = this.morseEncoder.generateAudio(text);

    // Resample to target rate (3200 Hz for model compatibility)
    const resampledAudio = this.resample(cleanAudio48k, 48000, mergedConfig.sampleRate);

    // Add noise if SNR is specified and not infinite (very high SNR = no noise)
    let finalAudio: Float32Array;
    if (mergedConfig.snrDb < 60) {
      finalAudio = this.addNoise(resampledAudio, mergedConfig.snrDb);
    } else {
      finalAudio = resampledAudio;
    }

    return {
      audio: finalAudio,
      label: text,
      config: { ...mergedConfig },
      duration: finalAudio.length / mergedConfig.sampleRate,
    };
  }

  /**
   * Generate a batch of random Morse code samples
   */
  generateRandom(
    length: number,
    charPool: string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
    config?: Partial<SyntheticConfig>,
  ): SyntheticResult[] {
    const results: SyntheticResult[] = [];

    for (let i = 0; i < length; i++) {
      // Random text of random length (3-15 characters)
      const textLength = Math.floor(Math.random() * 12) + 3;
      let text = "";
      for (let j = 0; j < textLength; j++) {
        text += charPool[Math.floor(Math.random() * charPool.length)];
      }
      // Ensure spaces don't appear at start/end and no double spaces
      text = text.trim().replace(/\s+/g, " ");

      if (text.length > 0) {
        results.push(this.generate(text, config));
      }
    }

    return results;
  }

  /**
   * Generate with WPM variation (for training diversity)
   */
  generateWithWPMVariation(
    text: string,
    wpmRange: [number, number] = [8, 50],
    config?: Partial<SyntheticConfig>,
  ): SyntheticResult {
    const wpm = wpmRange[0] + Math.random() * (wpmRange[1] - wpmRange[0]);
    return this.generate(text, { ...config, wpm });
  }

  /**
   * Generate with SNR variation (for robustness training)
   */
  generateWithSNRVariation(
    text: string,
    snrRange: [number, number] = [-10, 40],
    config?: Partial<SyntheticConfig>,
  ): SyntheticResult {
    const snrDb = snrRange[0] + Math.random() * (snrRange[1] - snrRange[0]);
    return this.generate(text, { ...config, snrDb });
  }

  /**
   * Generate with full variation (WPM, tone, SNR)
   */
  generateDiverse(
    text: string,
    config?: Partial<SyntheticConfig>,
  ): SyntheticResult {
    return this.generate(text, {
      ...config,
      wpm: config?.wpm ?? (8 + Math.random() * 42), // 8-50 WPM
      toneHz: config?.toneHz ?? (400 + Math.random() * 800), // 400-1200 Hz
      snrDb: config?.snrDb ?? (-10 + Math.random() * 50), // -10 to 40 dB
    });
  }
}

/** Default generator instance */
export const defaultGenerator = new SyntheticDataGenerator();
