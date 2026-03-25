/**
 * ggMorse-style CW/Morse Code Decoder
 *
 * Pure TypeScript implementation of real-time morse code decoder
 * using Goertzel algorithm for pitch detection.
 */

import { GoertzelFilter } from "./goertzel";
import { MorseDecoder } from "./morseDecoder";
import { SAMPLE_RATE, DECODABLE_MIN_FREQ_HZ, DECODABLE_MAX_FREQ_HZ } from "../const";

export interface GGMorseConfig {
  sampleRate?: number;
  minFrequency?: number;
  maxFrequency?: number;
  minWpm?: number;
  maxWpm?: number;
  noiseThreshold?: number;
  toneOnThreshold?: number;
}

export interface GGMorseResult {
  text: string;
  isActive: boolean;
  frequency: number | null;
  magnitude: number;
  confidence: number;
  morseBuffer: string;
  wpm: number;
}

const DEFAULT_CONFIG = {
  sampleRate: SAMPLE_RATE,
  minFrequency: DECODABLE_MIN_FREQ_HZ,
  maxFrequency: DECODABLE_MAX_FREQ_HZ,
  minWpm: 5,
  maxWpm: 55,
  noiseThreshold: 0.01,
  toneOnThreshold: 500,
};

export class GGMorse {
  private config: Required<GGMorseConfig>;
  private goertzelFilter: GoertzelFilter;
  private morseDecoder: MorseDecoder;
  private currentFrequency: number | null = null;
  private currentMagnitude: number = 0;
  private isToneActive: boolean = false;
  private toneOnsetSample: number = 0;
  private sampleCount: number = 0;
  private recentFrequencies: number[] = [];
  private wpmEstimate: number = 20;
  private decodedText: string = "";
  private onTextCallback?: (text: string) => void;
  private onToneCallback?: (isOn: boolean, frequency: number | null) => void;

  constructor(config: GGMorseConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    const centerFreq = (this.config.minFrequency + this.config.maxFrequency) / 2;
    const windowSize = Math.round(this.config.sampleRate / 10);

    this.goertzelFilter = new GoertzelFilter({
      sampleRate: this.config.sampleRate,
      targetFreq: centerFreq,
      windowSize,
    });

    this.morseDecoder = new MorseDecoder(this.config.sampleRate, this.wpmEstimate);
  }

  onText(callback: (text: string) => void): void {
    this.onTextCallback = callback;
  }

  onTone(callback: (isOn: boolean, frequency: number | null) => void): void {
    this.onToneCallback = callback;
  }

  processSamples(samples: Float32Array): void {
    // Process each sample through the Goertzel filter
    for (let i = 0; i < samples.length; i++) {
      this.goertzelFilter.processSample(samples[i]);
      this.sampleCount++;

      if (this.goertzelFilter.isWindowComplete() && i === samples.length - 1) {
        this.processWindow();
      }
    }
  }

  private processWindow(): void {
    const result = this.goertzelFilter.getResult(this.config.noiseThreshold);
    this.currentMagnitude = result.magnitude;

    if (result.isDetected && !this.isToneActive) {
      this.isToneActive = true;
      this.toneOnsetSample = this.sampleCount - this.goertzelFilter.currentSampleCount;
      this.morseDecoder.processTone(true, this.sampleCount);

      if (this.onToneCallback) {
        this.onToneCallback(true, this.currentFrequency);
      }
    } else if (!result.isDetected && this.isToneActive) {
      const toneDuration = this.sampleCount - this.toneOnsetSample;
      this.morseDecoder.processTone(false, this.sampleCount);
      this.isToneActive = false;
      this.updateWpmEstimate(toneDuration);

      if (this.onToneCallback) {
        this.onToneCallback(false, null);
      }
    } else if (this.isToneActive) {
      if (this.onToneCallback) {
        this.onToneCallback(true, this.currentFrequency);
      }
    }

    if (!this.isToneActive) {
      this.morseDecoder.checkForSpace(this.sampleCount);
      const currentText = this.morseDecoder.getText();
      if (currentText !== this.decodedText) {
        this.decodedText = currentText;
        if (this.onTextCallback) {
          this.onTextCallback(currentText);
        }
      }
    }

    this.goertzelFilter.reset();
  }

  private updateWpmEstimate(toneDuration: number): void {
    if (toneDuration > 0) {
      const unitMs = toneDuration / this.config.sampleRate * 1000;
      const estimatedWpm = Math.round(1200 / unitMs);
      this.wpmEstimate = Math.max(
        this.config.minWpm,
        Math.min(this.config.maxWpm, estimatedWpm)
      );
      this.morseDecoder.updateTiming(this.wpmEstimate);
    }
  }

  getResult(): GGMorseResult {
    return {
      text: this.decodedText,
      isActive: this.isToneActive,
      frequency: this.currentFrequency,
      magnitude: this.currentMagnitude,
      confidence: this.recentFrequencies.length > 0 ? 0.8 : 0,
      morseBuffer: this.morseDecoder.getState().currentMorse,
      wpm: this.wpmEstimate,
    };
  }

  flush(): string {
    const text = this.morseDecoder.flush();
    this.decodedText = text;
    return text;
  }

  reset(): void {
    this.goertzelFilter.reset();
    this.morseDecoder.reset();
    this.currentFrequency = null;
    this.currentMagnitude = 0;
    this.isToneActive = false;
    this.toneOnsetSample = 0;
    this.sampleCount = 0;
    this.recentFrequencies = [];
    this.decodedText = "";
  }

  get wpm(): number {
    return this.wpmEstimate;
  }

  setWpm(wpm: number): void {
    this.wpmEstimate = Math.max(this.config.minWpm, Math.min(this.config.maxWpm, wpm));
    this.morseDecoder.updateTiming(this.wpmEstimate);
  }
}
