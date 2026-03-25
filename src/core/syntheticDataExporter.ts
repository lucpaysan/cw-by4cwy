/**
 * Synthetic Data Exporter
 *
 * Export utilities for synthetic Morse code training data.
 * Supports JSONL and binary formats for Python training pipelines.
 */

import type { SyntheticConfig } from "./syntheticDataGenerator";

export interface ExportRecord {
  audio: Float32Array; // Raw audio samples
  text: string; // Ground truth label
  config: SyntheticConfig; // Generation parameters
  sampleRate: number;
  duration: number; // Duration in seconds
}

/**
 * Export records to JSONL format for Python consumption
 */
export function exportToJSONL(records: ExportRecord[]): string {
  const lines: string[] = [];

  for (const record of records) {
    const obj = {
      audio: Array.from(record.audio),
      text: record.text,
      config: record.config,
      sampleRate: record.sampleRate,
    };
    lines.push(JSON.stringify(obj));
  }

  return lines.join("\n");
}

/**
 * Export records to NDJSON (newline-delimited JSON) - same as JSONL
 */
export function exportToNDJSON(records: ExportRecord[]): string {
  return exportToJSONL(records);
}

/**
 * Export a single record to JSON string
 */
export function recordToJSON(record: ExportRecord): string {
  return JSON.stringify({
    audio: Array.from(record.audio),
    text: record.text,
    config: record.config,
    sampleRate: record.sampleRate,
  });
}

/**
 * Parse a JSONL/NDJSON record back from string
 */
export function parseJSONLRecord(line: string): ExportRecord | null {
  try {
    const obj = JSON.parse(line);
    const audio = new Float32Array(obj.audio);
    const sampleRate = obj.sampleRate ?? 3200;
    return {
      audio,
      text: obj.text,
      config: obj.config,
      sampleRate,
      duration: audio.length / sampleRate,
    };
  } catch {
    return null;
  }
}

/**
 * Parse multiple JSONL records
 */
export function parseJSONL(content: string): ExportRecord[] {
  const records: ExportRecord[] = [];
  const lines = content.split("\n");

  for (const line of lines) {
    if (line.trim()) {
      const record = parseJSONLRecord(line);
      if (record) {
        records.push(record);
      }
    }
  }

  return records;
}

/**
 * Download data as a file in the browser
 */
export function downloadData(data: string, filename: string, mimeType: string = "application/json"): void {
  const blob = new Blob([data], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

/**
 * Download synthetic dataset as JSONL
 */
export function downloadSyntheticDatasetJSONL(
  records: ExportRecord[],
  filename: string = "morse_synthetic_data.jsonl",
): void {
  const jsonl = exportToJSONL(records);
  downloadData(jsonl, filename, "application/x-jsonl");
}

/**
 * Download synthetic dataset as ZIP (using raw binary audio)
 * Note: This creates a simple format with separate audio and metadata files
 */
export function downloadSyntheticDatasetBinary(
  records: ExportRecord[],
  filename: string = "morse_synthetic_data",
): void {
  // Create a combined binary format:
  // [4 bytes: num_records]
  // For each record:
  //   [4 bytes: audio_length]
  //   [4 bytes: text_length]
  //   [audio_length * 4 bytes: audio data (float32)]
  //   [text_length * 2 bytes: text data (UTF-16)]
  //   [4 bytes each: wpm, toneHz, snrDb, sampleRate]

  const buffers: ArrayBuffer[] = [];

  // Number of records (4 bytes)
  const numRecords = new Uint32Array([records.length]);
  buffers.push(numRecords.buffer);

  for (const record of records) {
    const audio = record.audio;
    const text = record.text;

    // Audio length (4 bytes)
    const audioLen = new Uint32Array([audio.length]);
    buffers.push(audioLen.buffer);

    // Text length in UTF-16 chars (4 bytes)
    const textUtf16 = new TextEncoder().encode(text);
    const textLen = new Uint32Array([textUtf16.length]);
    buffers.push(textLen.buffer);

    // Audio data (float32 * length)
    buffers.push(audio.buffer);

    // Text data (UTF-8)
    buffers.push(textUtf16.buffer);

    // Config values (4 bytes each)
    const configData = new Float32Array([
      record.config.wpm,
      record.config.toneHz,
      record.config.snrDb,
      record.config.sampleRate,
      record.config.farnsworth,
    ]);
    buffers.push(configData.buffer);

    // Sample rate (4 bytes)
    const sampleRate = new Uint32Array([record.sampleRate]);
    buffers.push(sampleRate.buffer);
  }

  // Concatenate all buffers
  const totalLength = buffers.reduce((sum, buf) => sum + buf.byteLength, 0);
  const combined = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of buffers) {
    combined.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  const blob = new Blob([combined], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename + ".bin";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

/**
 * Create a metadata JSON file describing the dataset
 */
export function createDatasetMetadata(
  records: ExportRecord[],
  name: string = "morse_synthetic_dataset",
): object {
  // Calculate statistics
  const totalDuration = records.reduce((sum, r) => sum + r.duration, 0);
  const uniqueWpm = new Set(records.map(r => r.config.wpm));
  const uniqueSnr = new Set(records.map(r => r.config.snrDb));
  const uniqueTone = new Set(records.map(r => r.config.toneHz));

  return {
    name,
    version: "1.0",
    created: new Date().toISOString(),
    num_samples: records.length,
    total_duration_seconds: totalDuration,
    format: "jsonl",
    config: {
      wpm_range: [
        Math.min(...records.map(r => r.config.wpm)),
        Math.max(...records.map(r => r.config.wpm)),
      ],
      tone_hz_range: [
        Math.min(...records.map(r => r.config.toneHz)),
        Math.max(...records.map(r => r.config.toneHz)),
      ],
      snr_db_range: [
        Math.min(...records.map(r => r.config.snrDb)),
        Math.max(...records.map(r => r.config.snrDb)),
      ],
      sample_rate: records[0]?.sampleRate ?? 3200,
    },
    statistics: {
      unique_wpm_values: uniqueWpm.size,
      unique_snr_values: uniqueSnr.size,
      unique_tone_hz_values: uniqueTone.size,
      avg_duration_per_sample: totalDuration / records.length,
    },
  };
}

/**
 * Download dataset with metadata
 */
export function downloadSyntheticDataset(
  records: ExportRecord[],
  name: string = "morse_synthetic_dataset",
): void {
  // Download JSONL data
  downloadSyntheticDatasetJSONL(records, `${name}.jsonl`);

  // Download metadata
  const metadata = createDatasetMetadata(records, name);
  const metadataJson = JSON.stringify(metadata, null, 2);
  downloadData(metadataJson, `${name}_metadata.json`, "application/json");
}
