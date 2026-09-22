export interface GeneratedNote {
  column: number; // 0, 1, 2, 3
  startTime: number;
  endTime: number;
  isLn: boolean;
}

export interface BeatmapGenerationSettings {
  title: string;
  artist: string;
  creator: string;
  version: string;
  bpm: number;
  offsetMs: number;
  starRating: number;
  durationSec: number;
  lnProbability: number;
  svFluctuation: number;
}
