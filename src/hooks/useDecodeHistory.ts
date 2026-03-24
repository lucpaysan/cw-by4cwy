import { useEffect, useState } from "react";
import type { TextSegment } from "../utils/textDecoder";

interface DecodeHistoryEntry {
  text: string;
  timestamp: number;
  confirmed: boolean;
}

export const useDecodeHistory = (currentSegments: TextSegment[]) => {
  const [decodeHistory, setDecodeHistory] = useState<string>("");
  const [confirmedEntries, setConfirmedEntries] = useState<DecodeHistoryEntry[]>([]);

  const ZONE_WIDTH = 32;
  const ZONE_START_RATIO = 0.25;

  const clearHistory = () => {
    setDecodeHistory("");
    setConfirmedEntries([]);
  };

  useEffect(() => {
    const fullText = currentSegments.map((seg) => seg.text).join("");
    console.log(fullText.length)

    if (!fullText) {
      return;
    }

    // Calculate 32-char detection zone starting at 5% from the left
    const zoneStart = Math.max(0, Math.floor(fullText.length * ZONE_START_RATIO));
    const zoneEnd = Math.min(fullText.length, zoneStart + ZONE_WIDTH);
    const zoneText = fullText.slice(zoneStart, zoneEnd);

    // Extract only letters (A-Z, a-z) from the zone, no spaces or punctuation
    const lettersOnly = zoneText.replace(/[^A-Za-z0-9./?]/g, "");

    // Add zone content to history on every refresh, no comparison
    if (lettersOnly) {
      const newEntry: DecodeHistoryEntry = {
        text: lettersOnly,
        timestamp: Date.now(),
        confirmed: true,
      };

      setConfirmedEntries((prev) => [...prev, newEntry]);
      setDecodeHistory((prev) => prev + lettersOnly);
    }
  }, [currentSegments]);


  return {
    decodeHistory,
    confirmedEntries,
    currentText: "",
    clearHistory,
  };
};
