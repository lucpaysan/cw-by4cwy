import type { Tensor } from "onnxruntime-web";
import { ENGLISH_CONFIG } from "../const";

export type TextSegment = {
  text: string;
  isAbbreviation: boolean;
};

// CTC blank token index - "[UNK]" at vocabulary index 0
const CTC_BLANK_INDEX = 0;

/**
 * CTC Greedy Decode
 *
 * True CTC decoding rule:
 * - For each timestep, take argmax to get best class
 * - If class != blank AND class != last_emitted: emit this character
 * - Blank tokens are skipped entirely
 *
 * This handles "EEEE" (no blanks) -> single "E" naturally
 * because we only emit when a new character appears after a blank.
 *
 * Example:
 *   [E, E, BLANK, E, BLANK, T] -> "ET" (emit E after blank, emit T after blank)
 *   [E, E, E, E] -> "" (no blanks, so no emissions)
 *   [E, BLANK, E, E] -> "EE" (emit E after first blank, then E again without blank)
 */
function ctcGreedyDecode(
  pred: ArrayBuffer | Float32Array,
  predShape: readonly [number, number, number],
): string[] {
  const [batchSize, timeSteps, numClasses] = predShape;
  const vocabulary = ENGLISH_CONFIG.VOCABULARY;
  const results: string[] = [];

  // Convert to Float32Array if needed
  const predArray = pred instanceof Float32Array
    ? pred
    : new Float32Array(pred as unknown as ArrayBuffer);

  for (let b = 0; b < batchSize; b++) {
    let decoded = "";
    let prevEmittedIndex = CTC_BLANK_INDEX; // Track last emitted (non-blank) character

    for (let t = 0; t < timeSteps; t++) {
      // Find argmax at this timestep
      let maxProb = -Infinity;
      let maxIndex = CTC_BLANK_INDEX;
      const offset = b * timeSteps * numClasses + t * numClasses;

      for (let c = 0; c < numClasses; c++) {
        const prob = predArray[offset + c];
        if (prob > maxProb) {
          maxProb = prob;
          maxIndex = c;
        }
      }

      // CTC rule: emit if (current != blank) AND (current != last_emitted)
      if (maxIndex !== CTC_BLANK_INDEX && maxIndex !== prevEmittedIndex) {
        decoded += vocabulary[maxIndex] || "";
        prevEmittedIndex = maxIndex;
      }

      // Update prev even for blank (for next emission check)
      if (maxIndex === CTC_BLANK_INDEX) {
        // After blank, any character can be emitted
        prevEmittedIndex = CTC_BLANK_INDEX;
      }
    }

    results.push(decoded);
  }

  return results;
}

/**
 * CTC Beam Search Decode
 *
 * Explores multiple decoding paths using beam search.
 * Returns top-k decoded strings with their log probabilities.
 * Can find better results than greedy when model has ambiguous outputs.
 */
export interface BeamSearchResult {
  text: string;
  score: number; // log probability
}

export function ctcBeamSearchDecode(
  pred: ArrayBuffer | Float32Array,
  predShape: readonly [number, number, number],
  beamWidth: number = 10,
): BeamSearchResult[] {
  const [batchSize, timeSteps, numClasses] = predShape;
  const vocabulary = ENGLISH_CONFIG.VOCABULARY;
  const results: BeamSearchResult[] = [];

  // Convert to Float32Array if needed
  const predArray = pred instanceof Float32Array
    ? pred
    : new Float32Array(pred as unknown as ArrayBuffer);

  for (let b = 0; b < batchSize; b++) {
    // Each beam: {text, score, lastCharIndex}
    interface Beam {
      text: string;
      score: number;
      lastCharIndex: number;
    }

    let beams: Beam[] = [{ text: "", score: 0, lastCharIndex: CTC_BLANK_INDEX }];

    for (let t = 0; t < timeSteps; t++) {
      const offset = b * timeSteps * numClasses + t * numClasses;

      // Get probabilities for this timestep
      const probs = new Float32Array(numClasses);
      for (let c = 0; c < numClasses; c++) {
        probs[c] = predArray[offset + c];
      }

      // Log probabilities (numerically stable)
      const maxProb = Math.max(...Array.from(probs));
      const logProbs = Array.from(probs).map(p => Math.log(p + 1e-10) - maxProb);

      const newBeamsMap = new Map<string, Beam>();

      for (const beam of beams) {
        for (let c = 0; c < numClasses; c++) {
          const logP = logProbs[c];

          if (c === CTC_BLANK_INDEX) {
            // Extend with blank - doesn't change text, resets lastCharIndex
            const newBeam: Beam = {
              text: beam.text,
              score: beam.score + logP,
              lastCharIndex: CTC_BLANK_INDEX,
            };
            const key = `blank_${beam.text}_${beam.score}`;
            if (!newBeamsMap.has(key) || newBeamsMap.get(key)!.score < newBeam.score) {
              newBeamsMap.set(key, newBeam);
            }
          } else if (c === beam.lastCharIndex) {
            // Same as last emitted char (no emission, just extend)
            const newBeam: Beam = {
              text: beam.text,
              score: beam.score + logP,
              lastCharIndex: c,
            };
            const key = `same_${c}_${beam.text}_${beam.score}`;
            if (!newBeamsMap.has(key) || newBeamsMap.get(key)!.score < newBeam.score) {
              newBeamsMap.set(key, newBeam);
            }
          } else {
            // New character emitted
            const newText = beam.text + (vocabulary[c] || "");
            const newBeam: Beam = {
              text: newText,
              score: beam.score + logP,
              lastCharIndex: c,
            };
            // Deduplicate by text (keep highest score)
            if (!newBeamsMap.has(newText) || newBeamsMap.get(newText)!.score < newBeam.score) {
              newBeamsMap.set(newText, newBeam);
            }
          }
        }
      }

      // Keep top beamWidth beams
      beams = Array.from(newBeamsMap.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, beamWidth);
    }

    // Return best beam
    if (beams.length > 0) {
      // Sort by score (most negative = highest probability since log probs)
      beams.sort((a, b) => b.score - a.score);
      results.push({ text: beams[0].text, score: beams[0].score });
    } else {
      results.push({ text: "", score: 0 });
    }
  }

  return results;
}

function convertAbbreviationsWithSegments(str: string): TextSegment[] {
  const abbreviations = Object.entries(ENGLISH_CONFIG.ABBREVIATION) as [string, string][];

  if (abbreviations.length === 0) {
    return [{ text: str, isAbbreviation: false }];
  }

  const abbrevPattern = abbreviations.map(([abbrev]) => abbrev).join("|");
  const regex = new RegExp(`(${abbrevPattern})`, "g");

  const segments: TextSegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(str)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        text: str.slice(lastIndex, match.index),
        isAbbreviation: false,
      });
    }

    const matchedText = match[0];
    const abbrevEntry = abbreviations.find(([abbrev]) => abbrev === matchedText);
    const expansion = abbrevEntry ? abbrevEntry[1] : matchedText;

    segments.push({
      text: expansion,
      isAbbreviation: true,
    });

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < str.length) {
    segments.push({
      text: str.slice(lastIndex),
      isAbbreviation: false,
    });
  }

  return segments.length > 0 ? segments : [{ text: str, isAbbreviation: false }];
}

export function decodePredictions(
  pred: Tensor["data"],
  predShape: Tensor["dims"],
): TextSegment[][] {
  // Cast to Float32Array - ONNX returns float32 tensor data as Float32Array
  const predArray = pred as Float32Array;

  // Use CTC greedy decode for proper sequence-to-sequence decoding
  const decodedTexts = ctcGreedyDecode(predArray, predShape as readonly [number, number, number]);

  const outputSegments: TextSegment[][] = [];

  for (let i = 0; i < decodedTexts.length; i++) {
    const processedSegments = convertAbbreviationsWithSegments(decodedTexts[i]);
    outputSegments.push(processedSegments);
  }

  return outputSegments;
}
