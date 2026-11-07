# Motif library report

```
motif library motifs-v1: 1606 maps of 596 sets
  exact     338510 motifs (closed 321146), longest 48 chords, >= 8 chords: 65211
  lanes     521085 motifs (closed 515344), longest 48 chords, >= 8 chords: 303968
  rhythm    411578 motifs (closed 392871), longest 48 chords, >= 8 chords: 264278
  shape     523418 motifs (closed 511361), longest 48 chords, >= 8 chords: 289532
  hands     429353 motifs (closed 412169), longest 48 chords, >= 8 chords: 191164
  flow      406541 motifs (closed 395041), longest 48 chords, >= 8 chords: 209471
  ln        402463 motifs (closed 393449), longest 48 chords, >= 8 chords: 159921
  form: 9% of chords copy an earlier passage exactly, 9% mirrored, 8% as a variation (same rhythm, <= half of the lanes changed), 36% keep only the rhythm
```

## How maps repeat themselves (LZ77 parse of every map)

| stars | maps | exact copy | mirrored copy | variation (<= 50 % lanes changed) | same rhythm only | copy length (median beats) |
|---|---|---|---|---|---|---|
| 1-3* | 461 | 6% | 6% | 7% | 41% | <= 8 |
| 3-5* | 578 | 10% | 8% | 8% | 39% | <= 4 |
| 5-7* | 408 | 10% | 9% | 8% | 36% | <= 4 |
| 7*+ | 159 | 9% | 11% | 8% | 33% | <= 4 |

Copies start on a downbeat 11%, on a beat 19%, on a 1/2 27%, elsewhere 43%.

## Coverage: share of chords inside a known motif

| level | stars | >= 4 chords | >= 8 chords |
|---|---|---|---|
| exact | 1-3* | 82% | 19% |
| exact | 3-5* | 84% | 30% |
| exact | 5-7* | 80% | 26% |
| exact | 7*+ | 76% | 19% |
| lanes | 1-3* | 100% | 88% |
| lanes | 3-5* | 100% | 79% |
| lanes | 5-7* | 100% | 73% |
| lanes | 7*+ | 100% | 73% |
| rhythm | 1-3* | 99% | 85% |
| rhythm | 3-5* | 99% | 87% |
| rhythm | 5-7* | 99% | 82% |
| rhythm | 7*+ | 99% | 79% |
| shape | 1-3* | 100% | 86% |
| shape | 3-5* | 100% | 76% |
| shape | 5-7* | 100% | 71% |
| shape | 7*+ | 100% | 71% |
| hands | 1-3* | 97% | 60% |
| hands | 3-5* | 98% | 66% |
| hands | 5-7* | 97% | 60% |
| hands | 7*+ | 97% | 56% |
| flow | 1-3* | 98% | 73% |
| flow | 3-5* | 98% | 74% |
| flow | 5-7* | 97% | 66% |
| flow | 7*+ | 95% | 54% |
| ln | 1-3* | 94% | 56% |
| ln | 3-5* | 91% | 55% |
| ln | 5-7* | 86% | 45% |
| ln | 7*+ | 80% | 32% |

## Level `exact`

### 1-3*, 4-7 chords

**stream** - 4 chords, ~3.0 beats, 587 times in 111 sets (34% of maps at 1-3*); starts on downbeat 19%
```
|...#| 1
|#...| 1
|..#.| 1
|.#..| 1
```

**stream** - 4 chords, ~3.0 beats, 727 times in 112 sets (34% of maps at 1-3*); starts on downbeat 25%
```
|#...| 1
|..#.| 1
|.#..| 1
|...#| 1
```

**stairs** - 4 chords, ~3.0 beats, 485 times in 105 sets (32% of maps at 1-3*); starts on downbeat 46%
```
|#...| 1
|.#..| 1
|..#.| 1
|...#| 1
```

**stream** - 4 chords, ~3.0 beats, 593 times in 106 sets (32% of maps at 1-3*); starts on downbeat 25%
```
|.#..| 1
|..#.| 1
|#...| 1
|...#| 1
```

**stream** - 4 chords, ~1.5 beats, 1451 times in 197 sets (29% of maps at 1-3*); starts on downbeat 7%
```
|...#| 1/2
|#...| 1/2
|.#..| 1/2
|..#.| 1/2
```

**stream** - 4 chords, ~1.5 beats, 1969 times in 232 sets (29% of maps at 1-3*); starts on downbeat 12%
```
|#...| 1/2
|..#.| 1/2
|.#..| 1/2
|...#| 1/2
```

**stairs** - 4 chords, ~1.5 beats, 1631 times in 187 sets (29% of maps at 1-3*); starts on downbeat 17%
```
|#...| 1/2
|.#..| 1/2
|..#.| 1/2
|...#| 1/2
```

**stream** - 4 chords, ~1.5 beats, 1528 times in 221 sets (29% of maps at 1-3*); starts on downbeat 7%
```
|...#| 1/2
|#...| 1/2
|..#.| 1/2
|.#..| 1/2
```

### 1-3*, 8-15 chords

**jumpstream** - 8 chords, ~7.0 beats, 14 times in 11 sets (3% of maps at 1-3*); starts on downbeat 29%
```
|.#..| 1
|..##| 1
|#...| 1
|.##.| 1
|...#| 1
|##..| 1
|..#.| 1
|#..#| 1
```

**stream** - 8 chords, ~7.0 beats, 40 times in 19 sets (6% of maps at 1-3*); starts on downbeat 35%
```
|#...| 1
|..#.| 1
|.#..| 1
|...#| 1
|#...| 1
|..#.| 1
|.#..| 1
|...#| 1
```

**stream** - 8 chords, ~7.0 beats, 51 times in 20 sets (6% of maps at 1-3*); starts on downbeat 41%
```
|..#.| 1
|.#..| 1
|...#| 1
|#...| 1
|..#.| 1
|.#..| 1
|...#| 1
|#...| 1
```

**stream** - 8 chords, ~7.0 beats, 43 times in 21 sets (6% of maps at 1-3*); starts on downbeat 26%
```
|.#..| 1
|...#| 1
|#...| 1
|..#.| 1
|.#..| 1
|...#| 1
|#...| 1
|..#.| 1
```

**jumpstream** - 8 chords, ~7.0 beats, 14 times in 10 sets (3% of maps at 1-3*); starts on downbeat 7%
```
|#...| 1
|.##.| 1
|...#| 1
|##..| 1
|..#.| 1
|#..#| 1
|.#..| 1
|..##| 1
```

**jumpstream** - 8 chords, ~7.0 beats, 16 times in 9 sets (3% of maps at 1-3*); starts on downbeat 19%
```
|..##| 1
|#...| 1
|.##.| 1
|...#| 1
|##..| 1
|..#.| 1
|#..#| 1
|.#..| 1
```

**jumpstream** - 8 chords, ~7.0 beats, 12 times in 10 sets (3% of maps at 1-3*); starts on downbeat 42%
```
|#..#| 1
|.#..| 1
|..##| 1
|#...| 1
|.##.| 1
|...#| 1
|##..| 1
|..#.| 1
```

**stream** - 8 chords, ~7.0 beats, 36 times in 18 sets (5% of maps at 1-3*); starts on downbeat 17%
```
|...#| 1
|#...| 1
|..#.| 1
|.#..| 1
|...#| 1
|#...| 1
|..#.| 1
|.#..| 1
```

### 1-3*, 16+ chords

**stream** - 16 chords, ~15.0 beats, 3 times in 3 sets (1% of maps at 1-3*); starts on downbeat 33%
```
|..#.| 1
|.#..| 1
|#...| 1
|...#| 1
|..#.| 1
|.#..| 1
|#...| 1
|..#.| 1
|...#| 1
|.#..| 1
|..#.| 1
|#...| 1
|.#..| 1
|..#.| 1
|...#| 1
|.#..| 1
```

**stream** - 16 chords, ~15.0 beats, 4 times in 3 sets (1% of maps at 1-3*); starts on downbeat 50%
```
|...#| 1
|..#.| 1
|.#..| 1
|#...| 1
|..#.| 1
|...#| 1
|.#..| 1
|..#.| 1
|#...| 1
|.#..| 1
|..#.| 1
|...#| 1
|.#..| 1
|#...| 1
|..#.| 1
|.#..| 1
```

**jumpstream** - 16 chords, ~7.5 beats, 5 times in 3 sets (0% of maps at 1-3*); starts on downbeat 0%
```
|..##| 1/2
|.#..| 1/2
|#..#| 1/2
|..#.| 1/2
|##..| 1/2
|...#| 1/2
|#.#.| 1/2
|.#..| 1/2
|..##| 1/2
|#...| 1/2
|.#.#| 1/2
|..#.| 1/2
|##..| 1/2
|...#| 1/2
|#.#.| 1/2
|.#..| 1/2
```

**jumpstream** - 16 chords, ~7.5 beats, 6 times in 5 sets (0% of maps at 1-3*); starts on downbeat 33%
```
|.#..| 1/2
|..##| 1/2
|#...| 1/2
|.#.#| 1/2
|..#.| 1/2
|##..| 1/2
|...#| 1/2
|#.#.| 1/2
|.#..| 1/2
|..##| 1/2
|.#..| 1/2
|#..#| 1/2
|..#.| 1/2
|##..| 1/2
|..#.| 1/2
|#..#| 1/2
```

**stream** - 16 chords, ~7.5 beats, 3 times in 3 sets (1% of maps at 1-3*); starts on downbeat 0%
```
|..#.| 1/2
|#...| 1/2
|.#..| 1/2
|..#.| 1/2
|...#| 1/2
|.#..| 1/2
|..#.| 1/2
|#...| 1/2
|.#..| 1/2
|..#.| 1/2
|...#| 1/2
|.#..| 1/2
|#...| 1/2
|..#.| 1/2
|.#..| 1/2
|...#| 1/2
```

**jumpstream** - 17 chords, ~8.0 beats, 3 times in 3 sets (0% of maps at 1-3*); starts on downbeat 0%
```
|#...| 1/2
|..##| 1/2
|.#..| 1/2
|#..#| 1/2
|..#.| 1/2
|##..| 1/2
|..#.| 1/2
|#..#| 1/2
|.#..| 1/2
|..##| 1/2
|#...| 1/2
|.##.| 1/2
|...#| 1/2
|##..| 1/2
|..#.| 1/2
|#..#| 1/2
|.#..| 1/2
```

**stream** - 16 chords, ~7.5 beats, 6 times in 4 sets (1% of maps at 1-3*); starts on downbeat 0%
```
|...#| 1/2
|.#..| 1/2
|#...| 1/2
|..#.| 1/2
|.#..| 1/2
|...#| 1/2
|..#.| 1/2
|.#..| 1/2
|#...| 1/2
|..#.| 1/2
|...#| 1/2
|.#..| 1/2
|..#.| 1/2
|#...| 1/2
|.#..| 1/2
|..#.| 1/2
```

**stream** - 16 chords, ~7.5 beats, 13 times in 5 sets (1% of maps at 1-3*); starts on downbeat 0%
```
|.#..| 1/2
|...#| 1/2
|..#.| 1/2
|.#..| 1/2
|#...| 1/2
|..#.| 1/2
|.#..| 1/2
|...#| 1/2
|..#.| 1/2
|#...| 1/2
|.#..| 1/2
|..#.| 1/2
|...#| 1/2
|.#..| 1/2
|..#.| 1/2
|#...| 1/2
```

### 3-5*, 4-7 chords

**jumpstream** - 4 chords, ~0.8 beats, 3444 times in 348 sets (54% of maps at 3-5*); starts on downbeat 0%
```
|##..| 1/4
|...#| 1/4
|..#.| 1/4
|.#..| 1/4
```

**jumpstream** - 4 chords, ~0.8 beats, 4443 times in 359 sets (54% of maps at 3-5*); starts on downbeat 0%
```
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
```

**jumpstream** - 5 chords, ~1.0 beats, 2267 times in 270 sets (38% of maps at 3-5*); starts on downbeat 10%
```
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
```

**jumpstream** - 4 chords, ~0.8 beats, 3188 times in 328 sets (52% of maps at 3-5*); starts on downbeat 0%
```
|..##| 1/4
|.#..| 1/4
|#...| 1/4
|..#.| 1/4
```

**jumpstream** - 5 chords, ~1.0 beats, 1759 times in 262 sets (36% of maps at 3-5*); starts on downbeat 0%
```
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
```

**jumpstream** - 4 chords, ~0.7 beats, 3758 times in 345 sets (49% of maps at 3-5*); starts on downbeat 9%
```
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
```

**jumpstream** - 6 chords, ~1.3 beats, 1099 times in 197 sets (26% of maps at 3-5*); starts on downbeat 0%
```
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
```

**jumpstream** - 5 chords, ~1.0 beats, 1733 times in 248 sets (34% of maps at 3-5*); starts on downbeat 0%
```
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
```

### 3-5*, 8-15 chords

**jumpstream** - 8 chords, ~1.8 beats, 411 times in 103 sets (14% of maps at 3-5*); starts on downbeat 1%
```
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
```

**jumpstream** - 8 chords, ~1.8 beats, 454 times in 108 sets (13% of maps at 3-5*); starts on downbeat 0%
```
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
```

**jumpstream** - 8 chords, ~1.7 beats, 438 times in 105 sets (13% of maps at 3-5*); starts on downbeat 12%
```
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
```

**jumpstream** - 9 chords, ~2.0 beats, 320 times in 89 sets (12% of maps at 3-5*); starts on downbeat 1%
```
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
```

**jumpstream** - 8 chords, ~1.8 beats, 402 times in 104 sets (12% of maps at 3-5*); starts on downbeat 11%
```
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
```

**jumpstream** - 9 chords, ~2.0 beats, 346 times in 90 sets (11% of maps at 3-5*); starts on downbeat 0%
```
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
```

**jumpstream** - 8 chords, ~1.7 beats, 363 times in 87 sets (12% of maps at 3-5*); starts on downbeat 13%
```
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
```

**jumpstream** - 8 chords, ~1.7 beats, 398 times in 100 sets (12% of maps at 3-5*); starts on downbeat 0%
```
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
```

### 3-5*, 16+ chords

**jumpstream** - 16 chords, ~3.8 beats, 47 times in 20 sets (3% of maps at 3-5*); starts on downbeat 0%
```
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
```

**jumpstream** - 16 chords, ~3.7 beats, 43 times in 20 sets (2% of maps at 3-5*); starts on downbeat 0%
```
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
```

**jumpstream** - 16 chords, ~3.8 beats, 43 times in 19 sets (2% of maps at 3-5*); starts on downbeat 0%
```
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
```

**jumpstream** - 16 chords, ~3.8 beats, 37 times in 17 sets (2% of maps at 3-5*); starts on downbeat 35%
```
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
```

**jumpstream** - 17 chords, ~4.0 beats, 35 times in 16 sets (2% of maps at 3-5*); starts on downbeat 0%
```
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
```

**jumpstream** - 16 chords, ~3.8 beats, 38 times in 17 sets (2% of maps at 3-5*); starts on downbeat 18%
```
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
```

**jumpstream** - 17 chords, ~4.0 beats, 34 times in 15 sets (2% of maps at 3-5*); starts on downbeat 12%
```
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
```

**jumpstream** - 16 chords, ~3.7 beats, 38 times in 16 sets (2% of maps at 3-5*); starts on downbeat 16%
```
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
```

### 5-7*, 4-7 chords

**jumpstream** - 5 chords, ~1.0 beats, 1759 times in 262 sets (40% of maps at 5-7*); starts on downbeat 0%
```
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
```

**jumpstream** - 7 chords, ~1.5 beats, 839 times in 149 sets (24% of maps at 5-7*); starts on downbeat 0%
```
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
```

**jumpstream** - 6 chords, ~1.3 beats, 1102 times in 189 sets (30% of maps at 5-7*); starts on downbeat 0%
```
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
```

**jumpstream** - 6 chords, ~1.2 beats, 1031 times in 182 sets (30% of maps at 5-7*); starts on downbeat 9%
```
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
```

**jumpstream** - 5 chords, ~1.0 beats, 2267 times in 270 sets (39% of maps at 5-7*); starts on downbeat 10%
```
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
```

**jumpstream** - 4 chords, ~0.8 beats, 4443 times in 359 sets (53% of maps at 5-7*); starts on downbeat 0%
```
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
```

**stairs** - 4 chords, ~0.4 beats, 3144 times in 315 sets (52% of maps at 5-7*); starts on downbeat 1%
```
|#...| 1/8
|.#..| 1/8
|..#.| 1/8
|...#| 1/8
```

**jumpstream** - 5 chords, ~1.0 beats, 1733 times in 248 sets (37% of maps at 5-7*); starts on downbeat 0%
```
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
```

### 5-7*, 8-15 chords

**jumpstream** - 8 chords, ~1.8 beats, 454 times in 108 sets (14% of maps at 5-7*); starts on downbeat 0%
```
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
```

**jumpstream** - 8 chords, ~1.8 beats, 411 times in 103 sets (14% of maps at 5-7*); starts on downbeat 1%
```
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
```

**jumpstream** - 8 chords, ~1.8 beats, 402 times in 104 sets (14% of maps at 5-7*); starts on downbeat 11%
```
|..#.| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
```

**jumpstream** - 8 chords, ~1.7 beats, 438 times in 105 sets (13% of maps at 5-7*); starts on downbeat 12%
```
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
```

**jumpstream** - 8 chords, ~1.7 beats, 398 times in 100 sets (13% of maps at 5-7*); starts on downbeat 0%
```
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
```

**light jumpstream** - 8 chords, ~0.9 beats, 298 times in 92 sets (20% of maps at 5-7*); starts on downbeat 0%
```
|..##| 1/8
|#...| 1/8
|.#..| 1/8
|..#.| 1/8
|...#| 1/8
|#...| 1/8
|.#..| 1/8
|..#.| 1/8
```

**jumpstream** - 9 chords, ~2.0 beats, 346 times in 90 sets (11% of maps at 5-7*); starts on downbeat 0%
```
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
```

**jumpstream** - 9 chords, ~2.0 beats, 319 times in 83 sets (11% of maps at 5-7*); starts on downbeat 13%
```
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
```

### 5-7*, 16+ chords

**jumpstream** - 19 chords, ~4.5 beats, 14 times in 5 sets (3% of maps at 5-7*); starts on downbeat 0%
```
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
```

**jumpstream** - 25 chords, ~6.0 beats, 9 times in 3 sets (2% of maps at 5-7*); starts on downbeat 0%
```
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
```

**jumpstream** - 18 chords, ~4.3 beats, 16 times in 6 sets (3% of maps at 5-7*); starts on downbeat 0%
```
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
```

**jumpstream** - 24 chords, ~5.8 beats, 10 times in 4 sets (2% of maps at 5-7*); starts on downbeat 0%
```
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
```

**jumpstream** - 17 chords, ~4.0 beats, 17 times in 6 sets (3% of maps at 5-7*); starts on downbeat 0%
```
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
```

**jumpstream** - 23 chords, ~5.5 beats, 10 times in 3 sets (2% of maps at 5-7*); starts on downbeat 0%
```
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
```

**jumpstream** - 16 chords, ~3.8 beats, 21 times in 5 sets (3% of maps at 5-7*); starts on downbeat 0%
```
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
```

**jumpstream** - 22 chords, ~5.3 beats, 11 times in 5 sets (2% of maps at 5-7*); starts on downbeat 0%
```
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|##.#| 1/4
|..#.| 1/4
|.#.#| 1/4
|#...| 1/4
|.###| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#.##| 1/4
|.#..| 1/4
|#.#.| 1/4
|...#| 1/4
|###.| 1/4
|...#| 1/4
```

### 7*+, 4-7 chords

**stairs** - 4 chords, ~0.4 beats, 3144 times in 315 sets (61% of maps at 7*+); starts on downbeat 1%
```
|#...| 1/8
|.#..| 1/8
|..#.| 1/8
|...#| 1/8
```

**LN stairs** - 4 chords, ~0.4 beats, 633 times in 108 sets (56% of maps at 7*+); starts on downbeat 1%
```
|...O| 1/8
|..O.| 1/8
|.O..| 1/8
|O...| 1/8
```

**LN stream** - 4 chords, ~0.4 beats, 642 times in 102 sets (54% of maps at 7*+); starts on downbeat 0%
```
|.O..| 1/8
|..O.| 1/8
|...O| 1/8
|O...| 1/8
```

**LN jumpstream** - 5 chords, ~1.0 beats, 415 times in 89 sets (34% of maps at 7*+); starts on downbeat 0%
```
|.O..| 1/4
|O..O| 1/4
|..O.| 1/4
|OO..| 1/4
|...O| 1/4
```

**LN stream** - 4 chords, ~0.5 beats, 511 times in 103 sets (48% of maps at 7*+); starts on downbeat 1%
```
|O...| 1/6
|...O| 1/6
|.O..| 1/6
|..O.| 1/6
```

**LN stream** - 4 chords, ~0.5 beats, 512 times in 105 sets (48% of maps at 7*+); starts on downbeat 2%
```
|O...| 1/6
|..O.| 1/6
|.O..| 1/6
|...O| 1/6
```

**LN stream** - 4 chords, ~0.4 beats, 425 times in 72 sets (47% of maps at 7*+); starts on downbeat 1%
```
|..O.| 1/8
|.O..| 1/8
|...O| 1/8
|O...| 1/8
```

**LN stream** - 4 chords, ~0.4 beats, 748 times in 118 sets (47% of maps at 7*+); starts on downbeat 0%
```
|...O| 1/8
|O...| 1/8
|.O..| 1/8
|..O.| 1/8
```

### 7*+, 8-15 chords

**LN jumpstream** - 8 chords, ~1.8 beats, 79 times in 31 sets (15% of maps at 7*+); starts on downbeat 0%
```
|..OO| 1/4
|O...| 1/4
|.OO.| 1/4
|...O| 1/4
|OO..| 1/4
|..O.| 1/4
|O..O| 1/4
|.O..| 1/4
```

**chordjack mix** - 11 chords, ~2.5 beats, 23 times in 3 sets (19% of maps at 7*+); starts on downbeat 0%
```
|####| 1/4
|##..| 1/4
|####| 1/4
|..##| 1/4
|####| 1/4
|#..#| 1/4
|####| 1/4
|##..| 1/4
|####| 1/4
|.##.| 1/4
|####| 1/4
```

**chordjack mix** - 9 chords, ~2.0 beats, 34 times in 5 sets (21% of maps at 7*+); starts on downbeat 0%
```
|####| 1/4
|#.#.| 1/4
|####| 1/4
|.#.#| 1/4
|####| 1/4
|##..| 1/4
|####| 1/4
|..##| 1/4
|####| 1/4
```

**chordjack mix** - 8 chords, ~1.8 beats, 40 times in 5 sets (21% of maps at 7*+); starts on downbeat 0%
```
|.#.#| 1/4
|####| 1/4
|#.#.| 1/4
|####| 1/4
|..##| 1/4
|####| 1/4
|##..| 1/4
|####| 1/4
```

**LN jumpstream** - 8 chords, ~1.8 beats, 42 times in 23 sets (13% of maps at 7*+); starts on downbeat 0%
```
|.OO.| 1/4
|O...| 1/4
|..OO| 1/4
|.O..| 1/4
|O..O| 1/4
|..O.| 1/4
|OO..| 1/4
|...O| 1/4
```

**LN jumpstream** - 8 chords, ~1.8 beats, 74 times in 24 sets (13% of maps at 7*+); starts on downbeat 1%
```
|O..O| 1/4
|.O..| 1/4
|..OO| 1/4
|O...| 1/4
|.OO.| 1/4
|...O| 1/4
|OO..| 1/4
|..O.| 1/4
```

**chordjack mix** - 9 chords, ~2.0 beats, 43 times in 4 sets (19% of maps at 7*+); starts on downbeat 9%
```
|####| 1/4
|..##| 1/4
|####| 1/4
|##..| 1/4
|####| 1/4
|.##.| 1/4
|####| 1/4
|#.#.| 1/4
|####| 1/4
```

**chord jacks** - 8 chords, ~1.7 beats, 17 times in 3 sets (19% of maps at 7*+); starts on downbeat 0%
```
|####| 1/4
|###.| 1/4
|####| 1/4
|##.#| 1/4
|####| 1/4
|..##| 1/4
|####| 1/4
|##..| 1/4
```

### 7*+, 16+ chords

**LN jumpstream** - 17 chords, ~4.0 beats, 3 times in 3 sets (3% of maps at 7*+); starts on downbeat 0%
```
|.O..| 1/4
|..OO| 1/4
|O...| 1/4
|.OOO| 1/4
|O...| 1/4
|.OO.| 1/4
|...O| 1/4
|OOO.| 1/4
|...O| 1/4
|OO..| 1/4
|..O.| 1/4
|OO.O| 1/4
|..O.| 1/4
|O..O| 1/4
|.O..| 1/4
|O.OO| 1/4
|.O..| 1/4
```

**LN jumpstream** - 16 chords, ~3.8 beats, 3 times in 3 sets (3% of maps at 7*+); starts on downbeat 0%
```
|O..O| 1/4
|.O..| 1/4
|..OO| 1/4
|.O..| 1/4
|O.O.| 1/4
|...O| 1/4
|OO..| 1/4
|...O| 1/4
|.OO.| 1/4
|O...| 1/4
|..OO| 1/4
|O...| 1/4
|.O.O| 1/4
|..O.| 1/4
|OO..| 1/4
|..O.| 1/4
```

**jumpstream** - 19 chords, ~4.5 beats, 14 times in 5 sets (3% of maps at 7*+); starts on downbeat 0%
```
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
```

**LN light jumpstream** - 16 chords, ~2.0 beats, 3 times in 3 sets (5% of maps at 7*+); starts on downbeat 0%
```
|.O..| 1/4
|..OO| 1/8
|O...| 1/8
|.O..| 1/8
|..O.| 1/8
|...O| 1/8
|O...| 1/8
|.O..| 1/8
|..O.| 1/8
|...O| 1/8
|O...| 1/8
|.O..| 1/8
|..O.| 1/8
|...O| 1/8
|O...| 1/8
|.O..| 1/8
```

**jumpstream** - 18 chords, ~4.2 beats, 15 times in 6 sets (3% of maps at 7*+); starts on downbeat 0%
```
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
|.##.| 1/4
```

**LN light jumpstream** - 18 chords, ~4.2 beats, 5 times in 3 sets (3% of maps at 7*+); starts on downbeat 0%
```
|O...| 1/4
|..OO| 1/4
|.O..| 1/4
|O...| 1/4
|...O| 1/4
|O.O.| 1/4
|.O..| 1/4
|...O| 1/4
|O...| 1/4
|.O.O| 1/4
|..O.| 1/4
|O...| 1/4
|...O| 1/4
|OO..| 1/4
|..O.| 1/4
|...O| 1/4
|O...| 1/4
|.O.O| 1/4
```

**jumpstream** - 17 chords, ~4.0 beats, 21 times in 9 sets (3% of maps at 7*+); starts on downbeat 0%
```
|#...| 1/4
|.##.| 1/4
|...#| 1/4
|##..| 1/4
|..#.| 1/4
|#..#| 1/4
|..#.| 1/4
|##..| 1/4
|...#| 1/4
|.##.| 1/4
|#...| 1/4
|..##| 1/4
|.#..| 1/4
|#..#| 1/4
|.#..| 1/4
|..##| 1/4
|#...| 1/4
```

**LN light jumpstream** - 17 chords, ~4.0 beats, 5 times in 3 sets (3% of maps at 7*+); starts on downbeat 0%
```
|O...| 1/4
|...O| 1/4
|OO..| 1/4
|..O.| 1/4
|...O| 1/4
|O...| 1/4
|.O.O| 1/4
|..O.| 1/4
|O...| 1/4
|...O| 1/4
|O.O.| 1/4
|.O..| 1/4
|...O| 1/4
|O...| 1/4
|..OO| 1/4
|.O..| 1/4
|O...| 1/4
```


## Level `lanes`

### 1-3*, 4-7 chords

**stream** - 4 chords, ~1.1 beats, 27980 times in 581 sets (91% of maps at 1-3*); starts on downbeat 8%
```
|...#| 
|.#..| 
|..#.| 
|#...| 
```

**stream** - 4 chords, ~1.1 beats, 23257 times in 575 sets (91% of maps at 1-3*); starts on downbeat 5%
```
|#...| 
|...#| 
|.#..| 
|..#.| 
```

**stairs** - 4 chords, ~0.9 beats, 36579 times in 579 sets (91% of maps at 1-3*); starts on downbeat 10%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 4 chords, ~1.1 beats, 23880 times in 577 sets (90% of maps at 1-3*); starts on downbeat 7%
```
|.#..| 
|..#.| 
|#...| 
|...#| 
```

**stream** - 4 chords, ~0.9 beats, 28069 times in 580 sets (90% of maps at 1-3*); starts on downbeat 4%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 5 chords, ~1.6 beats, 9608 times in 539 sets (80% of maps at 1-3*); starts on downbeat 9%
```
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stream** - 4 chords, ~0.9 beats, 26513 times in 582 sets (89% of maps at 1-3*); starts on downbeat 5%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
```

**stream** - 4 chords, ~1.1 beats, 22100 times in 572 sets (88% of maps at 1-3*); starts on downbeat 6%
```
|..#.| 
|#...| 
|...#| 
|.#..| 
```

### 1-3*, 8-15 chords

**stream** - 8 chords, ~3.2 beats, 1570 times in 300 sets (38% of maps at 1-3*); starts on downbeat 10%
```
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
```

**stream** - 8 chords, ~3.4 beats, 1678 times in 319 sets (37% of maps at 1-3*); starts on downbeat 14%
```
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
```

**stream** - 8 chords, ~3.2 beats, 1583 times in 303 sets (37% of maps at 1-3*); starts on downbeat 11%
```
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
```

**stream** - 8 chords, ~3.3 beats, 1667 times in 300 sets (35% of maps at 1-3*); starts on downbeat 12%
```
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
```

**stream** - 8 chords, ~2.9 beats, 792 times in 272 sets (30% of maps at 1-3*); starts on downbeat 6%
```
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 9 chords, ~3.9 beats, 1086 times in 245 sets (28% of maps at 1-3*); starts on downbeat 12%
```
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
```

**stream** - 8 chords, ~2.7 beats, 867 times in 276 sets (28% of maps at 1-3*); starts on downbeat 14%
```
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 8 chords, ~2.7 beats, 1089 times in 268 sets (28% of maps at 1-3*); starts on downbeat 5%
```
|...#| 
|.#..| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
```

### 1-3*, 16+ chords

**stream** - 16 chords, ~8.6 beats, 135 times in 42 sets (4% of maps at 1-3*); starts on downbeat 22%
```
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
```

**stream** - 16 chords, ~9.1 beats, 140 times in 44 sets (3% of maps at 1-3*); starts on downbeat 20%
```
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
```

**stream** - 16 chords, ~8.6 beats, 141 times in 46 sets (3% of maps at 1-3*); starts on downbeat 23%
```
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
```

**stream** - 16 chords, ~8.4 beats, 139 times in 44 sets (3% of maps at 1-3*); starts on downbeat 14%
```
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
```

**stream** - 17 chords, ~8.8 beats, 107 times in 36 sets (3% of maps at 1-3*); starts on downbeat 18%
```
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stream** - 17 chords, ~9.8 beats, 121 times in 39 sets (3% of maps at 1-3*); starts on downbeat 20%
```
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
```

**stream** - 18 chords, ~9.6 beats, 88 times in 26 sets (3% of maps at 1-3*); starts on downbeat 16%
```
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
```

**stream** - 17 chords, ~8.8 beats, 109 times in 32 sets (3% of maps at 1-3*); starts on downbeat 23%
```
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
```

### 3-5*, 4-7 chords

**jumpstream** - 5 chords, ~1.4 beats, 4610 times in 483 sets (67% of maps at 3-5*); starts on downbeat 1%
```
|...#| 
|##..| 
|..#.| 
|#..#| 
|.#..| 
```

**jumpstream** - 5 chords, ~1.4 beats, 4960 times in 492 sets (67% of maps at 3-5*); starts on downbeat 1%
```
|..#.| 
|#..#| 
|.#..| 
|..##| 
|#...| 
```

**jumpstream** - 5 chords, ~1.3 beats, 6051 times in 456 sets (67% of maps at 3-5*); starts on downbeat 17%
```
|##..| 
|..#.| 
|#..#| 
|.#..| 
|..##| 
```

**jumpstream** - 5 chords, ~0.9 beats, 5696 times in 511 sets (67% of maps at 3-5*); starts on downbeat 5%
```
|##..| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stairs** - 4 chords, ~0.9 beats, 36579 times in 579 sets (93% of maps at 3-5*); starts on downbeat 10%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 5 chords, ~1.1 beats, 15241 times in 559 sets (82% of maps at 3-5*); starts on downbeat 4%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 4 chords, ~0.9 beats, 28069 times in 580 sets (91% of maps at 3-5*); starts on downbeat 4%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 4 chords, ~0.9 beats, 26513 times in 582 sets (91% of maps at 3-5*); starts on downbeat 5%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
```

### 3-5*, 8-15 chords

**jumpstream** - 8 chords, ~2.3 beats, 833 times in 205 sets (27% of maps at 3-5*); starts on downbeat 1%
```
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|..#.| 
```

**jumpstream** - 8 chords, ~2.2 beats, 853 times in 203 sets (24% of maps at 3-5*); starts on downbeat 21%
```
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
```

**jumpstream** - 8 chords, ~2.3 beats, 906 times in 213 sets (23% of maps at 3-5*); starts on downbeat 1%
```
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
```

**jumpstream** - 8 chords, ~2.3 beats, 801 times in 196 sets (23% of maps at 3-5*); starts on downbeat 22%
```
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|.#..| 
|..##| 
```

**jumpstream** - 8 chords, ~2.3 beats, 879 times in 207 sets (23% of maps at 3-5*); starts on downbeat 17%
```
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
```

**stream** - 8 chords, ~1.6 beats, 3959 times in 369 sets (44% of maps at 3-5*); starts on downbeat 16%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**jumpstream** - 9 chords, ~2.6 beats, 596 times in 160 sets (20% of maps at 3-5*); starts on downbeat 1%
```
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|.#..| 
```

**jumpstream** - 9 chords, ~2.6 beats, 685 times in 178 sets (20% of maps at 3-5*); starts on downbeat 1%
```
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
```

### 3-5*, 16+ chords

**stream** - 16 chords, ~2.8 beats, 735 times in 104 sets (10% of maps at 3-5*); starts on downbeat 24%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 16 chords, ~2.9 beats, 638 times in 93 sets (10% of maps at 3-5*); starts on downbeat 2%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 16 chords, ~2.8 beats, 606 times in 93 sets (9% of maps at 3-5*); starts on downbeat 3%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
```

**stream** - 17 chords, ~3.0 beats, 571 times in 83 sets (9% of maps at 3-5*); starts on downbeat 2%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 17 chords, ~3.0 beats, 582 times in 84 sets (8% of maps at 3-5*); starts on downbeat 22%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 16 chords, ~2.9 beats, 562 times in 85 sets (8% of maps at 3-5*); starts on downbeat 2%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**stream** - 17 chords, ~2.9 beats, 523 times in 76 sets (7% of maps at 3-5*); starts on downbeat 2%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**stream** - 18 chords, ~3.0 beats, 521 times in 73 sets (7% of maps at 3-5*); starts on downbeat 20%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

### 5-7*, 4-7 chords

**jumpstream** - 5 chords, ~0.9 beats, 5696 times in 511 sets (74% of maps at 5-7*); starts on downbeat 5%
```
|##..| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**jumpstream** - 5 chords, ~1.4 beats, 4960 times in 492 sets (70% of maps at 5-7*); starts on downbeat 1%
```
|..#.| 
|#..#| 
|.#..| 
|..##| 
|#...| 
```

**jumpstream** - 5 chords, ~1.4 beats, 4610 times in 483 sets (70% of maps at 5-7*); starts on downbeat 1%
```
|...#| 
|##..| 
|..#.| 
|#..#| 
|.#..| 
```

**jumpstream** - 7 chords, ~2.0 beats, 1704 times in 291 sets (42% of maps at 5-7*); starts on downbeat 1%
```
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
```

**stream** - 5 chords, ~1.1 beats, 15241 times in 559 sets (85% of maps at 5-7*); starts on downbeat 4%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**jumpstream** - 6 chords, ~1.7 beats, 2597 times in 383 sets (51% of maps at 5-7*); starts on downbeat 2%
```
|..##| 
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
```

**stream** - 6 chords, ~1.3 beats, 9489 times in 513 sets (77% of maps at 5-7*); starts on downbeat 4%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 5 chords, ~1.0 beats, 13818 times in 553 sets (84% of maps at 5-7*); starts on downbeat 3%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
```

### 5-7*, 8-15 chords

**jumpstream** - 8 chords, ~2.3 beats, 906 times in 213 sets (27% of maps at 5-7*); starts on downbeat 1%
```
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
```

**jumpstream** - 8 chords, ~2.3 beats, 879 times in 207 sets (27% of maps at 5-7*); starts on downbeat 17%
```
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
```

**jumpstream** - 8 chords, ~2.3 beats, 780 times in 193 sets (26% of maps at 5-7*); starts on downbeat 1%
```
|..##| 
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
```

**stream** - 8 chords, ~1.6 beats, 3959 times in 369 sets (52% of maps at 5-7*); starts on downbeat 16%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**jumpstream** - 8 chords, ~2.3 beats, 833 times in 205 sets (25% of maps at 5-7*); starts on downbeat 1%
```
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|..#.| 
```

**jumpstream** - 8 chords, ~2.2 beats, 853 times in 203 sets (25% of maps at 5-7*); starts on downbeat 21%
```
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
```

**stream** - 8 chords, ~1.5 beats, 3115 times in 328 sets (48% of maps at 5-7*); starts on downbeat 3%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
```

**stream** - 8 chords, ~1.6 beats, 3094 times in 337 sets (47% of maps at 5-7*); starts on downbeat 3%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

### 5-7*, 16+ chords

**stream** - 16 chords, ~2.8 beats, 735 times in 104 sets (17% of maps at 5-7*); starts on downbeat 24%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 16 chords, ~2.9 beats, 638 times in 93 sets (15% of maps at 5-7*); starts on downbeat 2%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 16 chords, ~2.8 beats, 606 times in 93 sets (15% of maps at 5-7*); starts on downbeat 3%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
```

**stream** - 17 chords, ~3.0 beats, 582 times in 84 sets (14% of maps at 5-7*); starts on downbeat 22%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 17 chords, ~3.0 beats, 571 times in 83 sets (14% of maps at 5-7*); starts on downbeat 2%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 18 chords, ~3.0 beats, 521 times in 73 sets (13% of maps at 5-7*); starts on downbeat 20%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 16 chords, ~2.9 beats, 562 times in 85 sets (13% of maps at 5-7*); starts on downbeat 2%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**stream** - 17 chords, ~3.0 beats, 517 times in 79 sets (13% of maps at 5-7*); starts on downbeat 3%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
```

### 7*+, 4-7 chords

**jumpstream** - 5 chords, ~0.9 beats, 5696 times in 511 sets (76% of maps at 7*+); starts on downbeat 5%
```
|##..| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**jumpstream** - 5 chords, ~1.4 beats, 4610 times in 483 sets (75% of maps at 7*+); starts on downbeat 1%
```
|...#| 
|##..| 
|..#.| 
|#..#| 
|.#..| 
```

**light jumpstream** - 6 chords, ~1.4 beats, 1322 times in 362 sets (67% of maps at 7*+); starts on downbeat 3%
```
|##..| 
|..#.| 
|...#| 
|.#..| 
|..#.| 
|#...| 
```

**stream** - 6 chords, ~1.3 beats, 9489 times in 513 sets (84% of maps at 7*+); starts on downbeat 4%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**light jumpstream** - 6 chords, ~1.1 beats, 1970 times in 383 sets (64% of maps at 7*+); starts on downbeat 1%
```
|..##| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
```

**stream** - 7 chords, ~1.5 beats, 5517 times in 431 sets (74% of maps at 7*+); starts on downbeat 3%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**jumpstream** - 5 chords, ~1.0 beats, 3961 times in 461 sets (69% of maps at 7*+); starts on downbeat 28%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|..##| 
```

**stream** - 6 chords, ~1.5 beats, 2275 times in 404 sets (79% of maps at 7*+); starts on downbeat 5%
```
|#...| 
|...#| 
|..#.| 
|#...| 
|.#..| 
|..#.| 
```

### 7*+, 8-15 chords

**stream** - 8 chords, ~1.6 beats, 3959 times in 369 sets (62% of maps at 7*+); starts on downbeat 16%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 8 chords, ~2.2 beats, 1028 times in 293 sets (62% of maps at 7*+); starts on downbeat 11%
```
|..#.| 
|...#| 
|.#..| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|..#.| 
```

**stream** - 8 chords, ~1.9 beats, 1351 times in 287 sets (61% of maps at 7*+); starts on downbeat 6%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**stream** - 8 chords, ~1.8 beats, 835 times in 242 sets (59% of maps at 7*+); starts on downbeat 4%
```
|#...| 
|.#..| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
```

**stream** - 8 chords, ~1.9 beats, 723 times in 235 sets (59% of maps at 7*+); starts on downbeat 8%
```
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 8 chords, ~1.8 beats, 770 times in 242 sets (59% of maps at 7*+); starts on downbeat 8%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 8 chords, ~1.9 beats, 1332 times in 265 sets (59% of maps at 7*+); starts on downbeat 6%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 8 chords, ~2.0 beats, 927 times in 260 sets (59% of maps at 7*+); starts on downbeat 4%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|..#.| 
|.#..| 
```

### 7*+, 16+ chords

**stream** - 16 chords, ~3.2 beats, 31 times in 22 sets (23% of maps at 7*+); starts on downbeat 6%
```
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 17 chords, ~3.5 beats, 30 times in 21 sets (21% of maps at 7*+); starts on downbeat 7%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 16 chords, ~2.9 beats, 187 times in 36 sets (21% of maps at 7*+); starts on downbeat 7%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 19 chords, ~3.8 beats, 18 times in 14 sets (18% of maps at 7*+); starts on downbeat 6%
```
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 18 chords, ~4.0 beats, 21 times in 15 sets (18% of maps at 7*+); starts on downbeat 10%
```
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 18 chords, ~3.1 beats, 138 times in 26 sets (18% of maps at 7*+); starts on downbeat 6%
```
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 17 chords, ~3.0 beats, 143 times in 26 sets (18% of maps at 7*+); starts on downbeat 3%
```
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
```

**stream** - 17 chords, ~3.0 beats, 142 times in 27 sets (18% of maps at 7*+); starts on downbeat 6%
```
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```


## Level `rhythm`

### 1-3*, 4-7 chords

**motif** - 4 chords, ~1.5 beats, 5474 times in 382 sets (67% of maps at 1-3*); starts on downbeat 21%
```
1x1 1/2x1 1/2x1 1/2x1
```

**motif** - 4 chords, ~2.0 beats, 2157 times in 253 sets (44% of maps at 1-3*); starts on downbeat 7%
```
1/2x1 1/2x1 1/2x2 1x1
```

**motif** - 4 chords, ~2.0 beats, 2461 times in 264 sets (44% of maps at 1-3*); starts on downbeat 0%
```
1/2x1 1/2x2 1x1 1/2x1
```

**motif** - 5 chords, ~2.0 beats, 3574 times in 337 sets (59% of maps at 1-3*); starts on downbeat 26%
```
1x1 1/2x1 1/2x1 1/2x1 1/2x1
```

**motif** - 4 chords, ~2.0 beats, 1679 times in 215 sets (43% of maps at 1-3*); starts on downbeat 9%
```
1x1 1/2x1 1/2x2 1x1
```

**motif** - 5 chords, ~2.5 beats, 1645 times in 229 sets (39% of maps at 1-3*); starts on downbeat 0%
```
1/2x1 1/2x1 1/2x1 1/2x2 1x1
```

**motif** - 4 chords, ~2.0 beats, 3127 times in 307 sets (64% of maps at 1-3*); starts on downbeat 15%
```
1x1 1x1 1/2x1 1/2x1
```

**motif** - 4 chords, ~2.5 beats, 2524 times in 277 sets (62% of maps at 1-3*); starts on downbeat 15%
```
1x1 1x1 1x1 1/2x1
```

### 1-3*, 8-15 chords

**motif** - 8 chords, ~7.0 beats, 552 times in 128 sets (36% of maps at 1-3*); starts on downbeat 5%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2
```

**motif** - 12 chords, ~11.0 beats, 276 times in 91 sets (27% of maps at 1-3*); starts on downbeat 8%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2
```

**motif** - 8 chords, ~6.5 beats, 498 times in 125 sets (32% of maps at 1-3*); starts on downbeat 18%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1/2x1
```

**motif** - 15 chords, ~14.0 beats, 219 times in 80 sets (23% of maps at 1-3*); starts on downbeat 6%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2
```

**motif** - 8 chords, ~6.0 beats, 494 times in 129 sets (32% of maps at 1-3*); starts on downbeat 18%
```
1x1 1x1 1x1 1x1 1x1 1x1 1/2x1 1/2x1
```

**motif** - 13 chords, ~12.0 beats, 249 times in 86 sets (25% of maps at 1-3*); starts on downbeat 72%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2
```

**motif** - 11 chords, ~10.0 beats, 284 times in 92 sets (27% of maps at 1-3*); starts on downbeat 9%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2
```

**motif** - 14 chords, ~13.0 beats, 236 times in 82 sets (24% of maps at 1-3*); starts on downbeat 7%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2
```

### 1-3*, 16+ chords

**motif** - 16 chords, ~15.0 beats, 203 times in 77 sets (22% of maps at 1-3*); starts on downbeat 5%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2
```

**motif** - 16 chords, ~15.0 beats, 161 times in 58 sets (18% of maps at 1-3*); starts on downbeat 86%
```
1x2 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1
```

**motif** - 16 chords, ~15.0 beats, 141 times in 57 sets (17% of maps at 1-3*); starts on downbeat 7%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2 1x1
```

**motif** - 17 chords, ~16.0 beats, 134 times in 56 sets (16% of maps at 1-3*); starts on downbeat 3%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2 1x1
```

**motif** - 18 chords, ~17.0 beats, 119 times in 48 sets (14% of maps at 1-3*); starts on downbeat 5%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2 1x1 1x1 1x1
```

**motif** - 16 chords, ~15.0 beats, 117 times in 45 sets (14% of maps at 1-3*); starts on downbeat 9%
```
1x1 1x2 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1
```

**motif** - 19 chords, ~18.0 beats, 114 times in 47 sets (13% of maps at 1-3*); starts on downbeat 4%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2 1x1 1x1 1x1
```

**motif** - 17 chords, ~16.0 beats, 121 times in 48 sets (14% of maps at 1-3*); starts on downbeat 5%
```
1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x1 1x2 1x1 1x1
```

### 3-5*, 4-7 chords

**motif** - 4 chords, ~1.3 beats, 5432 times in 427 sets (60% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/2x1 1/2x2
```

**motif** - 4 chords, ~1.0 beats, 5212 times in 408 sets (57% of maps at 3-5*); starts on downbeat 27%
```
1/2x2 1/2x1 1/4x1 1/4x2
```

**motif** - 5 chords, ~1.3 beats, 6352 times in 463 sets (67% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/2x1
```

**motif** - 5 chords, ~1.0 beats, 4991 times in 429 sets (62% of maps at 3-5*); starts on downbeat 32%
```
1/2x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 4 chords, ~1.0 beats, 6811 times in 472 sets (69% of maps at 3-5*); starts on downbeat 1%
```
1/4x1 1/4x1 1/4x2 1/2x1
```

**motif** - 5 chords, ~1.8 beats, 3301 times in 348 sets (45% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/2x1 1/2x2 1/2x1
```

**motif** - 7 chords, ~1.5 beats, 27692 times in 505 sets (74% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 5 chords, ~1.3 beats, 3721 times in 402 sets (58% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/2x2
```

### 3-5*, 8-15 chords

**motif** - 8 chords, ~1.8 beats, 21014 times in 467 sets (67% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 8 chords, ~1.8 beats, 5164 times in 457 sets (66% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x1 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 9 chords, ~2.0 beats, 18229 times in 437 sets (61% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1
```

**motif** - 9 chords, ~2.0 beats, 18157 times in 430 sets (60% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 8 chords, ~1.7 beats, 19310 times in 449 sets (63% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1
```

**motif** - 10 chords, ~2.3 beats, 16280 times in 413 sets (57% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 11 chords, ~2.5 beats, 14748 times in 399 sets (54% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 10 chords, ~2.3 beats, 15562 times in 414 sets (56% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1
```

### 3-5*, 16+ chords

**motif** - 16 chords, ~3.8 beats, 9438 times in 322 sets (41% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 17 chords, ~4.0 beats, 8576 times in 299 sets (37% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 18 chords, ~4.3 beats, 8072 times in 288 sets (35% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 16 chords, ~3.7 beats, 8760 times in 303 sets (37% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1
```

**motif** - 17 chords, ~4.0 beats, 8669 times in 298 sets (36% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1
```

**motif** - 16 chords, ~3.7 beats, 8677 times in 301 sets (37% of maps at 3-5*); starts on downbeat 24%
```
1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 17 chords, ~4.0 beats, 8172 times in 291 sets (36% of maps at 3-5*); starts on downbeat 23%
```
1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 19 chords, ~4.5 beats, 7495 times in 270 sets (33% of maps at 3-5*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1
```

### 5-7*, 4-7 chords

**motif** - 4 chords, ~0.8 beats, 10787 times in 484 sets (78% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3
```

**motif** - 6 chords, ~1.3 beats, 3647 times in 398 sets (61% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x3
```

**motif** - 5 chords, ~1.0 beats, 4287 times in 425 sets (64% of maps at 5-7*); starts on downbeat 12%
```
1/4x2 1/4x1 1/4x2 1/4x1 1/4x3
```

**motif** - 6 chords, ~1.3 beats, 3654 times in 392 sets (58% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x3
```

**motif** - 4 chords, ~0.7 beats, 4674 times in 464 sets (68% of maps at 5-7*); starts on downbeat 16%
```
1/2x2 1/4x1 1/4x2 1/4x1
```

**motif** - 4 chords, ~1.0 beats, 5212 times in 408 sets (51% of maps at 5-7*); starts on downbeat 27%
```
1/2x2 1/2x1 1/4x1 1/4x2
```

**motif** - 5 chords, ~1.0 beats, 4002 times in 401 sets (60% of maps at 5-7*); starts on downbeat 25%
```
1/4x2 1/4x1 1/4x1 1/4x1 1/4x3
```

**motif** - 7 chords, ~1.5 beats, 31782 times in 482 sets (76% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1
```

### 5-7*, 8-15 chords

**motif** - 8 chords, ~1.8 beats, 2112 times in 306 sets (47% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x3
```

**motif** - 14 chords, ~3.3 beats, 12617 times in 338 sets (53% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2
```

**motif** - 13 chords, ~3.0 beats, 14077 times in 350 sets (55% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1
```

**motif** - 15 chords, ~3.5 beats, 11711 times in 325 sets (51% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1
```

**motif** - 12 chords, ~2.8 beats, 15341 times in 364 sets (57% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2
```

**motif** - 8 chords, ~1.8 beats, 24913 times in 444 sets (69% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2
```

**motif** - 11 chords, ~2.5 beats, 17481 times in 379 sets (59% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1
```

**motif** - 9 chords, ~2.0 beats, 22189 times in 427 sets (65% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1
```

### 5-7*, 16+ chords

**motif** - 16 chords, ~3.8 beats, 10527 times in 291 sets (46% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2
```

**motif** - 17 chords, ~4.0 beats, 9858 times in 262 sets (41% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1
```

**motif** - 16 chords, ~3.8 beats, 9438 times in 322 sets (42% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 16 chords, ~3.7 beats, 10063 times in 263 sets (41% of maps at 5-7*); starts on downbeat 13%
```
1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1
```

**motif** - 17 chords, ~4.0 beats, 8576 times in 299 sets (40% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 16 chords, ~3.7 beats, 8760 times in 303 sets (41% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1
```

**motif** - 18 chords, ~4.3 beats, 8072 times in 288 sets (38% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 18 chords, ~4.3 beats, 9124 times in 244 sets (38% of maps at 5-7*); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2
```

### 7*+, 4-7 chords

**motif** - 4 chords, ~0.8 beats, 10787 times in 484 sets (90% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3
```

**motif** - 5 chords, ~0.8 beats, 1974 times in 250 sets (59% of maps at 7*+); starts on downbeat 0%
```
1/6x1 1/6x1 1/6x2 1/4x1 1/4x2
```

**motif** - 5 chords, ~1.0 beats, 7250 times in 373 sets (78% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3 1/4x1
```

**motif** - 6 chords, ~1.3 beats, 4950 times in 293 sets (70% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2
```

**motif** - 5 chords, ~1.0 beats, 6667 times in 344 sets (77% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x3 1/4x1 1/4x2 1/4x1
```

**motif** - 7 chords, ~1.5 beats, 4376 times in 263 sets (64% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1
```

**motif** - 4 chords, ~0.7 beats, 8423 times in 395 sets (83% of maps at 7*+); starts on downbeat 2%
```
1/4x2 1/4x1 1/4x3 1/4x1
```

**motif** - 4 chords, ~0.7 beats, 7546 times in 361 sets (82% of maps at 7*+); starts on downbeat 25%
```
1/4x3 1/4x1 1/4x2 1/4x1
```

### 7*+, 8-15 chords

**motif** - 8 chords, ~1.8 beats, 2393 times in 168 sets (57% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1 1/4x3
```

**motif** - 8 chords, ~1.8 beats, 2112 times in 306 sets (54% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x2 1/4x1 1/4x3
```

**motif** - 8 chords, ~1.7 beats, 2331 times in 174 sets (54% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x3 1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2
```

**motif** - 11 chords, ~2.5 beats, 1846 times in 126 sets (45% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1
```

**motif** - 9 chords, ~2.0 beats, 2185 times in 164 sets (49% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x3 1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1
```

**motif** - 10 chords, ~2.3 beats, 1942 times in 135 sets (45% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2
```

**motif** - 10 chords, ~2.3 beats, 1900 times in 130 sets (45% of maps at 7*+); starts on downbeat 1%
```
1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1 1/4x3 1/4x1 1/4x2 1/4x1
```

**motif** - 8 chords, ~1.8 beats, 2443 times in 317 sets (51% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x3
```

### 7*+, 16+ chords

**motif** - 20 chords, ~4.8 beats, 7361 times in 269 sets (43% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 18 chords, ~4.3 beats, 8072 times in 288 sets (45% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 19 chords, ~4.5 beats, 7382 times in 270 sets (43% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 19 chords, ~4.5 beats, 7795 times in 279 sets (43% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 17 chords, ~4.0 beats, 8576 times in 299 sets (45% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```

**motif** - 17 chords, ~4.0 beats, 8669 times in 298 sets (45% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1
```

**motif** - 17 chords, ~4.0 beats, 8172 times in 291 sets (45% of maps at 7*+); starts on downbeat 23%
```
1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2
```

**motif** - 18 chords, ~4.3 beats, 7817 times in 280 sets (43% of maps at 7*+); starts on downbeat 0%
```
1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1 1/4x2 1/4x1 1/4x1 1/4x1
```


## Level `shape`

### 1-3*, 4-7 chords

**trill / zigzag** - 4 chords, ~1.1 beats, 22705 times in 581 sets (89% of maps at 1-3*); starts on downbeat 6%
```
1< 1>far 1<far 1>
```

**motif** - 4 chords, ~0.9 beats, 20320 times in 564 sets (84% of maps at 1-3*); starts on downbeat 8%
```
1<far 1> 1>far 1<
```

**trill / zigzag** - 5 chords, ~1.4 beats, 10660 times in 546 sets (73% of maps at 1-3*); starts on downbeat 4%
```
1< 1< 1>far 1<far 1>
```

**motif** - 5 chords, ~1.3 beats, 8175 times in 532 sets (72% of maps at 1-3*); starts on downbeat 4%
```
1> 1<far 1> 1>far 1<
```

**motif** - 5 chords, ~1.3 beats, 10569 times in 545 sets (71% of maps at 1-3*); starts on downbeat 5%
```
1< 1>far 1<far 1> 1>
```

**trill / zigzag** - 4 chords, ~1.1 beats, 8159 times in 565 sets (75% of maps at 1-3*); starts on downbeat 4%
```
1< 1> 1<far 1>far
```

**motif** - 5 chords, ~1.3 beats, 7726 times in 533 sets (67% of maps at 1-3*); starts on downbeat 9%
```
1<far 1> 1>far 1< 1<
```

**trill / zigzag** - 5 chords, ~1.6 beats, 4768 times in 500 sets (66% of maps at 1-3*); starts on downbeat 8%
```
1< 1>far 1<far 1> 1<far
```

### 1-3*, 8-15 chords

**trill / zigzag** - 8 chords, ~3.4 beats, 1430 times in 295 sets (36% of maps at 1-3*); starts on downbeat 12%
```
1<far 1>far 1<far 1> 1<far 1>far 1<far 1>
```

**trill / zigzag** - 8 chords, ~3.3 beats, 1268 times in 288 sets (34% of maps at 1-3*); starts on downbeat 9%
```
1> 1<far 1>far 1<far 1> 1<far 1>far 1<far
```

**motif** - 8 chords, ~2.6 beats, 778 times in 257 sets (26% of maps at 1-3*); starts on downbeat 14%
```
1<far 1> 1> 1> 1<far 1>far 1< 1>far
```

**trill / zigzag** - 8 chords, ~3.3 beats, 661 times in 236 sets (25% of maps at 1-3*); starts on downbeat 13%
```
1< 1>far 1<far 1> 1<far 1>far 1<far 1>
```

**motif** - 8 chords, ~2.2 beats, 1772 times in 326 sets (25% of maps at 1-3*); starts on downbeat 3%
```
1> 1> 1<far 1>far 1< 1< 1>far 1<far
```

**motif** - 8 chords, ~2.7 beats, 782 times in 270 sets (25% of maps at 1-3*); starts on downbeat 12%
```
1>far 1<far 1>far 1< 1< 1>far 1<far 1>
```

**motif** - 8 chords, ~2.1 beats, 1989 times in 349 sets (25% of maps at 1-3*); starts on downbeat 4%
```
1< 1>far 1< 1<far 1> 1> 1> 1<far
```

**motif** - 8 chords, ~2.1 beats, 1829 times in 324 sets (25% of maps at 1-3*); starts on downbeat 5%
```
1> 1<far 1>far 1< 1< 1>far 1<far 1>
```

### 1-3*, 16+ chords

**trill / zigzag** - 16 chords, ~8.2 beats, 115 times in 40 sets (4% of maps at 1-3*); starts on downbeat 21%
```
1>far 1< 1>far 1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1<far
```

**trill / zigzag** - 16 chords, ~9.1 beats, 129 times in 41 sets (3% of maps at 1-3*); starts on downbeat 19%
```
1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far
```

**trill / zigzag** - 16 chords, ~8.3 beats, 114 times in 35 sets (3% of maps at 1-3*); starts on downbeat 10%
```
1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far
```

**trill / zigzag** - 17 chords, ~9.1 beats, 93 times in 29 sets (3% of maps at 1-3*); starts on downbeat 22%
```
1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far
```

**trill / zigzag** - 17 chords, ~8.6 beats, 101 times in 35 sets (3% of maps at 1-3*); starts on downbeat 19%
```
1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1<far
```

**trill / zigzag** - 18 chords, ~9.6 beats, 82 times in 26 sets (3% of maps at 1-3*); starts on downbeat 17%
```
1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far 1> 1<far 1>far 1<far
```

**trill / zigzag** - 16 chords, ~8.1 beats, 127 times in 42 sets (3% of maps at 1-3*); starts on downbeat 20%
```
1>far 1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1<far 1>far 1<
```

**motif** - 16 chords, ~13.2 beats, 12 times in 9 sets (2% of maps at 1-3*); starts on downbeat 8%
```
1<far 1>far 1< 1>far 1<far 1>far 1< 1>far 1< 1<far 1>far 1<far 1> 1<far 1>far 1<far
```

### 3-5*, 4-7 chords

**motif** - 6 chords, ~1.7 beats, 4635 times in 426 sets (59% of maps at 3-5*); starts on downbeat 17%
```
2<far 1>far 2< 1< 2>far 1<far
```

**motif** - 5 chords, ~1.3 beats, 5629 times in 448 sets (65% of maps at 3-5*); starts on downbeat 17%
```
2<far 1>far 2< 1< 2>far
```

**trill / zigzag** - 4 chords, ~1.0 beats, 11534 times in 572 sets (90% of maps at 3-5*); starts on downbeat 1%
```
1< 2>far 1<far 1>
```

**motif** - 5 chords, ~1.4 beats, 5173 times in 450 sets (63% of maps at 3-5*); starts on downbeat 1%
```
1>far 2< 1< 2>far 1<far
```

**motif** - 4 chords, ~0.9 beats, 20320 times in 564 sets (86% of maps at 3-5*); starts on downbeat 8%
```
1<far 1> 1>far 1<
```

**trill / zigzag** - 4 chords, ~1.1 beats, 22705 times in 581 sets (85% of maps at 3-5*); starts on downbeat 6%
```
1< 1>far 1<far 1>
```

**motif** - 4 chords, ~0.9 beats, 11109 times in 564 sets (85% of maps at 3-5*); starts on downbeat 34%
```
2<far 1>far 1> 1<far
```

**trill / zigzag** - 4 chords, ~1.0 beats, 10717 times in 566 sets (84% of maps at 3-5*); starts on downbeat 26%
```
2<far 1>far 1<far 1>
```

### 3-5*, 8-15 chords

**motif** - 8 chords, ~2.2 beats, 937 times in 208 sets (23% of maps at 3-5*); starts on downbeat 19%
```
2>far 1<far 2> 1> 2<far 1>far 2< 1<
```

**motif** - 8 chords, ~2.3 beats, 767 times in 185 sets (21% of maps at 3-5*); starts on downbeat 15%
```
2> 1> 2<far 1>far 2< 1< 2>far 1<far
```

**motif** - 8 chords, ~2.3 beats, 910 times in 228 sets (27% of maps at 3-5*); starts on downbeat 2%
```
1>far 2<far 1>far 2< 1< 2>far 1<far 2>far
```

**motif** - 8 chords, ~2.2 beats, 733 times in 167 sets (20% of maps at 3-5*); starts on downbeat 0%
```
1>far 2< 1< 2>far 1<far 2> 1> 2<far
```

**motif** - 9 chords, ~2.5 beats, 683 times in 160 sets (18% of maps at 3-5*); starts on downbeat 20%
```
2>far 1<far 2> 1> 2<far 1>far 2< 1< 2>far
```

**motif** - 9 chords, ~2.5 beats, 621 times in 150 sets (17% of maps at 3-5*); starts on downbeat 0%
```
1<far 2> 1> 2<far 1>far 2< 1< 2>far 1<far
```

**motif** - 8 chords, ~2.0 beats, 1782 times in 333 sets (36% of maps at 3-5*); starts on downbeat 8%
```
1>far 1< 1<far 1> 1> 1> 1<far 1>
```

**motif** - 8 chords, ~2.1 beats, 1989 times in 349 sets (36% of maps at 3-5*); starts on downbeat 4%
```
1< 1>far 1< 1<far 1> 1> 1> 1<far
```

### 3-5*, 16+ chords

**trill / zigzag** - 16 chords, ~4.1 beats, 195 times in 94 sets (8% of maps at 3-5*); starts on downbeat 22%
```
2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 1>far
```

**motif** - 16 chords, ~2.7 beats, 607 times in 62 sets (6% of maps at 3-5*); starts on downbeat 18%
```
1>far 1< 1<far 1> 1>far 1< 1<far 1> 1>far 1< 1<far 1> 1>far 1< 1<far 1>
```

**trill / zigzag** - 17 chords, ~4.5 beats, 156 times in 81 sets (7% of maps at 3-5*); starts on downbeat 32%
```
2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 1<far
```

**trill / zigzag** - 16 chords, ~3.8 beats, 2186 times in 109 sets (10% of maps at 3-5*); starts on downbeat 11%
```
2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far
```

**motif** - 16 chords, ~4.3 beats, 67 times in 29 sets (3% of maps at 3-5*); starts on downbeat 1%
```
1>far 2<far 1<far 2>far 1<far 2> 1> 2<far 1>far 2<far 1<far 2>far 1<far 2> 1> 2<far
```

**motif** - 16 chords, ~2.7 beats, 699 times in 100 sets (10% of maps at 3-5*); starts on downbeat 23%
```
1>far 1< 1< 1< 1>far 1< 1< 1< 1>far 1< 1< 1< 1>far 1< 1< 1<
```

**motif** - 16 chords, ~4.2 beats, 63 times in 28 sets (3% of maps at 3-5*); starts on downbeat 44%
```
2<far 1>far 2<far 1<far 2>far 1<far 2> 1> 2<far 1>far 2<far 1<far 2>far 1<far 2> 1>
```

**motif** - 16 chords, ~4.3 beats, 64 times in 25 sets (3% of maps at 3-5*); starts on downbeat 22%
```
2>far 1<far 2> 1> 2<far 1>far 2<far 1<far 2>far 1<far 2> 1> 2<far 1>far 2<far 1<far
```

### 5-7*, 4-7 chords

**motif** - 6 chords, ~1.7 beats, 4635 times in 426 sets (59% of maps at 5-7*); starts on downbeat 17%
```
2<far 1>far 2< 1< 2>far 1<far
```

**motif** - 5 chords, ~1.3 beats, 5629 times in 448 sets (64% of maps at 5-7*); starts on downbeat 17%
```
2<far 1>far 2< 1< 2>far
```

**motif** - 5 chords, ~1.4 beats, 5173 times in 450 sets (64% of maps at 5-7*); starts on downbeat 1%
```
1>far 2< 1< 2>far 1<far
```

**motif** - 7 chords, ~2.0 beats, 1978 times in 328 sets (44% of maps at 5-7*); starts on downbeat 2%
```
1<far 2>far 1<far 2> 1> 2<far 1>far
```

**motif** - 5 chords, ~1.3 beats, 4170 times in 454 sets (62% of maps at 5-7*); starts on downbeat 1%
```
1>far 2<far 1= 2>far 1<far
```

**trill / zigzag** - 4 chords, ~1.0 beats, 11534 times in 572 sets (85% of maps at 5-7*); starts on downbeat 1%
```
1< 2>far 1<far 1>
```

**trill / zigzag** - 4 chords, ~1.1 beats, 22705 times in 581 sets (85% of maps at 5-7*); starts on downbeat 6%
```
1< 1>far 1<far 1>
```

**trill / zigzag** - 5 chords, ~1.4 beats, 10660 times in 546 sets (75% of maps at 5-7*); starts on downbeat 4%
```
1< 1< 1>far 1<far 1>
```

### 5-7*, 8-15 chords

**motif** - 8 chords, ~2.1 beats, 1989 times in 349 sets (46% of maps at 5-7*); starts on downbeat 4%
```
1< 1>far 1< 1<far 1> 1> 1> 1<far
```

**motif** - 8 chords, ~2.2 beats, 937 times in 208 sets (22% of maps at 5-7*); starts on downbeat 19%
```
2>far 1<far 2> 1> 2<far 1>far 2< 1<
```

**motif** - 8 chords, ~1.9 beats, 1647 times in 329 sets (44% of maps at 5-7*); starts on downbeat 9%
```
1<far 1> 1> 1> 1<far 1< 1>far 1<
```

**motif** - 8 chords, ~2.1 beats, 1994 times in 348 sets (42% of maps at 5-7*); starts on downbeat 10%
```
1>far 1< 1< 1>far 1<far 1> 1> 1<far
```

**motif** - 8 chords, ~2.3 beats, 873 times in 204 sets (28% of maps at 5-7*); starts on downbeat 17%
```
2<far 1>far 2<far 1<far 2>far 1<far 2> 1>
```

**motif** - 8 chords, ~2.3 beats, 910 times in 228 sets (28% of maps at 5-7*); starts on downbeat 2%
```
1>far 2<far 1>far 2< 1< 2>far 1<far 2>far
```

**motif** - 8 chords, ~2.0 beats, 1805 times in 322 sets (41% of maps at 5-7*); starts on downbeat 9%
```
1<far 1>far 1< 1< 1>far 1<far 1> 1>
```

**motif** - 8 chords, ~2.0 beats, 1782 times in 333 sets (39% of maps at 5-7*); starts on downbeat 8%
```
1>far 1< 1<far 1> 1> 1> 1<far 1>
```

### 5-7*, 16+ chords

**trill / zigzag** - 16 chords, ~4.1 beats, 195 times in 94 sets (16% of maps at 5-7*); starts on downbeat 22%
```
2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 1>far
```

**trill / zigzag** - 17 chords, ~4.5 beats, 156 times in 81 sets (15% of maps at 5-7*); starts on downbeat 32%
```
2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 1<far
```

**trill / zigzag** - 16 chords, ~3.8 beats, 2186 times in 109 sets (20% of maps at 5-7*); starts on downbeat 11%
```
2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far
```

**trill / zigzag** - 18 chords, ~4.9 beats, 106 times in 59 sets (12% of maps at 5-7*); starts on downbeat 32%
```
2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 1<far
```

**trill / zigzag** - 16 chords, ~4.2 beats, 90 times in 52 sets (10% of maps at 5-7*); starts on downbeat 9%
```
2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 1>far 1<
```

**trill / zigzag** - 17 chords, ~4.1 beats, 1946 times in 84 sets (18% of maps at 5-7*); starts on downbeat 10%
```
2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far
```

**trill / zigzag** - 17 chords, ~4.1 beats, 78 times in 47 sets (9% of maps at 5-7*); starts on downbeat 0%
```
2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 1>far 1<
```

**trill / zigzag** - 18 chords, ~4.4 beats, 74 times in 44 sets (8% of maps at 5-7*); starts on downbeat 55%
```
2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 2<far 2>far 1<far 1>
```

### 7*+, 4-7 chords

**motif** - 5 chords, ~1.4 beats, 2655 times in 440 sets (70% of maps at 7*+); starts on downbeat 1%
```
1> 2< 1< 2>far 1<far
```

**motif** - 6 chords, ~1.7 beats, 4635 times in 426 sets (51% of maps at 7*+); starts on downbeat 17%
```
2<far 1>far 2< 1< 2>far 1<far
```

**motif** - 7 chords, ~1.9 beats, 3491 times in 437 sets (71% of maps at 7*+); starts on downbeat 10%
```
1>far 1< 1<far 1> 1> 1> 1<far
```

**motif** - 5 chords, ~1.3 beats, 5355 times in 504 sets (82% of maps at 7*+); starts on downbeat 6%
```
1< 1< 1>far 1> 1<far
```

**motif** - 7 chords, ~1.8 beats, 2459 times in 394 sets (69% of maps at 7*+); starts on downbeat 3%
```
1> 1<far 1> 1>far 1< 1< 1<
```

**motif** - 7 chords, ~1.8 beats, 3083 times in 420 sets (69% of maps at 7*+); starts on downbeat 3%
```
1> 1> 1<far 1>far 1< 1< 1>far
```

**motif** - 7 chords, ~1.8 beats, 3407 times in 430 sets (69% of maps at 7*+); starts on downbeat 9%
```
1<far 1> 1> 1<far 1>far 1< 1<
```

**motif** - 6 chords, ~1.7 beats, 1309 times in 365 sets (74% of maps at 7*+); starts on downbeat 4%
```
1> 1< 1>far 1<far 1> 1>
```

### 7*+, 8-15 chords

**motif** - 8 chords, ~2.1 beats, 1994 times in 348 sets (67% of maps at 7*+); starts on downbeat 10%
```
1>far 1< 1< 1>far 1<far 1> 1> 1<far
```

**motif** - 8 chords, ~2.1 beats, 1989 times in 349 sets (66% of maps at 7*+); starts on downbeat 4%
```
1< 1>far 1< 1<far 1> 1> 1> 1<far
```

**motif** - 9 chords, ~2.2 beats, 1281 times in 246 sets (61% of maps at 7*+); starts on downbeat 3%
```
1< 1>far 1<far 1> 1> 1<far 1>far 1< 1<
```

**motif** - 9 chords, ~2.4 beats, 1305 times in 267 sets (59% of maps at 7*+); starts on downbeat 12%
```
1<far 1> 1> 1<far 1>far 1< 1< 1>far 1<far
```

**motif** - 9 chords, ~2.3 beats, 1264 times in 252 sets (59% of maps at 7*+); starts on downbeat 9%
```
1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far
```

**motif** - 8 chords, ~2.2 beats, 1772 times in 326 sets (62% of maps at 7*+); starts on downbeat 3%
```
1> 1> 1<far 1>far 1< 1< 1>far 1<far
```

**motif** - 8 chords, ~2.1 beats, 1068 times in 279 sets (62% of maps at 7*+); starts on downbeat 3%
```
1< 1< 1>far 1<far 1> 1> 1> 1<far
```

**motif** - 8 chords, ~1.9 beats, 821 times in 249 sets (62% of maps at 7*+); starts on downbeat 8%
```
1<far 1> 1> 1<far 1>far 1< 1< 1<
```

### 7*+, 16+ chords

**motif** - 16 chords, ~3.3 beats, 243 times in 53 sets (21% of maps at 7*+); starts on downbeat 3%
```
1> 1<far 1>far 1< 1< 1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1>
```

**motif** - 16 chords, ~3.1 beats, 234 times in 47 sets (21% of maps at 7*+); starts on downbeat 9%
```
1<far 1>far 1< 1< 1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1> 1>
```

**motif** - 16 chords, ~3.3 beats, 31 times in 22 sets (21% of maps at 7*+); starts on downbeat 16%
```
1> 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far
```

**motif** - 17 chords, ~3.3 beats, 219 times in 44 sets (19% of maps at 7*+); starts on downbeat 1%
```
1> 1> 1<far 1>far 1< 1< 1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1>
```

**motif** - 16 chords, ~3.2 beats, 238 times in 49 sets (19% of maps at 7*+); starts on downbeat 1%
```
1< 1< 1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1> 1> 1<far 1>far
```

**motif** - 18 chords, ~3.8 beats, 21 times in 16 sets (18% of maps at 7*+); starts on downbeat 24%
```
1> 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1>
```

**motif** - 17 chords, ~4.9 beats, 9 times in 4 sets (8% of maps at 7*+); starts on downbeat 11%
```
3> 1>o 3< 1<faro 3>far 1=o 3< 1<o 3> 1>faro 3<far 1=o 3> 1>o 3< 1<faro 3>far
```

**motif** - 17 chords, ~3.9 beats, 24 times in 17 sets (18% of maps at 7*+); starts on downbeat 21%
```
1> 1> 1> 1<far 1>far 1< 1< 1>far 1<far 1> 1> 1<far 1>far 1< 1< 1>far 1<far
```


## Level `hands`

### 1-3*, 4-7 chords

**motif** - 4 chords, ~2.0 beats, 931 times in 178 sets (45% of maps at 1-3*); starts on downbeat 14%
```
1:R1 1:L1 1/2:L1 1/2:R1
```

**motif** - 4 chords, ~2.0 beats, 845 times in 190 sets (41% of maps at 1-3*); starts on downbeat 16%
```
1:L1 1:R1 1/2:L1 1/2:R1
```

**motif** - 4 chords, ~2.0 beats, 907 times in 170 sets (41% of maps at 1-3*); starts on downbeat 14%
```
1:R1 1/2:L1 1/2:R1 1:L1
```

**motif** - 4 chords, ~2.5 beats, 759 times in 169 sets (39% of maps at 1-3*); starts on downbeat 3%
```
1/2:L1 1/2:R1 1:L1 1:R1
```

**motif** - 4 chords, ~1.5 beats, 1216 times in 224 sets (49% of maps at 1-3*); starts on downbeat 26%
```
1:L1 1/2:L1 1/2:R1 1/2:R1
```

**motif** - 4 chords, ~2.5 beats, 540 times in 145 sets (36% of maps at 1-3*); starts on downbeat 2%
```
1/2:L1 1/2:R1 1:R1 1:L1
```

**motif** - 4 chords, ~1.5 beats, 3115 times in 328 sets (47% of maps at 1-3*); starts on downbeat 0%
```
1/2:R1 1/2:L1 1/2:L1 1/2:R2
```

**motif** - 4 chords, ~1.5 beats, 1561 times in 264 sets (46% of maps at 1-3*); starts on downbeat 21%
```
1:L1 1/2:R1 1/2:L1 1/2:R1
```

### 1-3*, 8-15 chords

**motif** - 8 chords, ~7.0 beats, 341 times in 91 sets (28% of maps at 1-3*); starts on downbeat 15%
```
1:L1 1:R1 1:L1 1:R1 1:L1 1:R1 1:R1 1:L1
```

**motif** - 8 chords, ~7.0 beats, 669 times in 92 sets (28% of maps at 1-3*); starts on downbeat 15%
```
1:L1 1:R1 1:R1 1:L1 1:L1 1:R1 1:R1 1:L1
```

**motif** - 8 chords, ~7.0 beats, 1088 times in 90 sets (27% of maps at 1-3*); starts on downbeat 27%
```
1:R1 1:L1 1:R1 1:L1 1:R1 1:L1 1:R1 1:L1
```

**motif** - 8 chords, ~3.5 beats, 1630 times in 175 sets (11% of maps at 1-3*); starts on downbeat 0%
```
1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2
```

**motif** - 8 chords, ~7.0 beats, 309 times in 88 sets (26% of maps at 1-3*); starts on downbeat 20%
```
1:L1 1:R1 1:L1 1:R1 1:R1 1:L1 1:L1 1:R1
```

**motif** - 8 chords, ~7.0 beats, 141 times in 40 sets (10% of maps at 1-3*); starts on downbeat 22%
```
1:R2 1:L1 1:B2 1:R1 1:L2 1:R1 1:B2 1:L1
```

**motif** - 8 chords, ~7.0 beats, 145 times in 43 sets (10% of maps at 1-3*); starts on downbeat 26%
```
1:L1 1:R2 1:L1 1:B2 1:R1 1:L2 1:R1 1:B2
```

**motif** - 8 chords, ~3.5 beats, 1430 times in 169 sets (10% of maps at 1-3*); starts on downbeat 0%
```
1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2
```

### 1-3*, 16+ chords

**motif** - 16 chords, ~7.5 beats, 309 times in 42 sets (3% of maps at 1-3*); starts on downbeat 0%
```
1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2
```

**motif** - 16 chords, ~7.5 beats, 329 times in 45 sets (2% of maps at 1-3*); starts on downbeat 0%
```
1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2
```

**motif** - 17 chords, ~8.0 beats, 284 times in 40 sets (2% of maps at 1-3*); starts on downbeat 0%
```
1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1
```

**motif** - 18 chords, ~8.5 beats, 236 times in 31 sets (2% of maps at 1-3*); starts on downbeat 0%
```
1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2
```

**motif** - 16 chords, ~7.5 beats, 295 times in 41 sets (2% of maps at 1-3*); starts on downbeat 28%
```
1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1
```

**motif** - 19 chords, ~9.0 beats, 204 times in 27 sets (2% of maps at 1-3*); starts on downbeat 21%
```
1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2
```

**motif** - 17 chords, ~8.0 beats, 289 times in 42 sets (2% of maps at 1-3*); starts on downbeat 21%
```
1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2
```

**motif** - 17 chords, ~8.0 beats, 242 times in 31 sets (2% of maps at 1-3*); starts on downbeat 29%
```
1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2 1/2:R1 1/2:B2 1/2:L1 1/2:R2 1/2:L1 1/2:B2 1/2:R1 1/2:L2
```

### 3-5*, 4-7 chords

**motif** - 6 chords, ~1.3 beats, 8359 times in 414 sets (54% of maps at 3-5*); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 7 chords, ~1.5 beats, 7142 times in 379 sets (49% of maps at 3-5*); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 5 chords, ~1.0 beats, 9299 times in 434 sets (57% of maps at 3-5*); starts on downbeat 9%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 6 chords, ~1.2 beats, 7647 times in 396 sets (51% of maps at 3-5*); starts on downbeat 10%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 4 chords, ~0.8 beats, 17994 times in 507 sets (72% of maps at 3-5*); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 7 chords, ~1.5 beats, 5577 times in 358 sets (44% of maps at 3-5*); starts on downbeat 9%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 5 chords, ~1.0 beats, 13772 times in 471 sets (64% of maps at 3-5*); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 5 chords, ~1.0 beats, 13585 times in 470 sets (64% of maps at 3-5*); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

### 3-5*, 8-15 chords

**motif** - 8 chords, ~1.8 beats, 5077 times in 336 sets (40% of maps at 3-5*); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 8 chords, ~1.8 beats, 4645 times in 322 sets (40% of maps at 3-5*); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2
```

**motif** - 8 chords, ~1.7 beats, 4838 times in 329 sets (39% of maps at 3-5*); starts on downbeat 10%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 9 chords, ~2.0 beats, 4158 times in 307 sets (37% of maps at 3-5*); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```

**motif** - 8 chords, ~1.7 beats, 4367 times in 318 sets (38% of maps at 3-5*); starts on downbeat 13%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 9 chords, ~2.0 beats, 4028 times in 296 sets (35% of maps at 3-5*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 9 chords, ~2.0 beats, 3909 times in 297 sets (34% of maps at 3-5*); starts on downbeat 10%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 10 chords, ~2.2 beats, 3217 times in 271 sets (31% of maps at 3-5*); starts on downbeat 10%
```
1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```

### 3-5*, 16+ chords

**motif** - 16 chords, ~3.8 beats, 1255 times in 137 sets (15% of maps at 3-5*); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2
```

**motif** - 16 chords, ~3.7 beats, 1283 times in 137 sets (13% of maps at 3-5*); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

**motif** - 16 chords, ~3.7 beats, 1213 times in 129 sets (13% of maps at 3-5*); starts on downbeat 12%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 17 chords, ~4.0 beats, 1141 times in 118 sets (12% of maps at 3-5*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 17 chords, ~4.0 beats, 1156 times in 125 sets (12% of maps at 3-5*); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```

**motif** - 16 chords, ~3.7 beats, 1189 times in 125 sets (12% of maps at 3-5*); starts on downbeat 17%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 18 chords, ~4.3 beats, 1006 times in 112 sets (11% of maps at 3-5*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 17 chords, ~4.0 beats, 1060 times in 118 sets (11% of maps at 3-5*); starts on downbeat 11%
```
1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

### 5-7*, 4-7 chords

**motif** - 7 chords, ~1.5 beats, 7142 times in 379 sets (59% of maps at 5-7*); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 6 chords, ~1.3 beats, 8359 times in 414 sets (63% of maps at 5-7*); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 5 chords, ~1.0 beats, 9299 times in 434 sets (68% of maps at 5-7*); starts on downbeat 9%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 6 chords, ~1.2 beats, 7647 times in 396 sets (61% of maps at 5-7*); starts on downbeat 10%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 7 chords, ~1.5 beats, 5577 times in 358 sets (56% of maps at 5-7*); starts on downbeat 9%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 7 chords, ~1.5 beats, 5391 times in 346 sets (54% of maps at 5-7*); starts on downbeat 12%
```
1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

**motif** - 7 chords, ~1.5 beats, 7133 times in 396 sets (60% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```

**motif** - 6 chords, ~1.3 beats, 9159 times in 427 sets (65% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

### 5-7*, 8-15 chords

**motif** - 8 chords, ~1.7 beats, 4838 times in 329 sets (52% of maps at 5-7*); starts on downbeat 10%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 8 chords, ~1.8 beats, 5077 times in 336 sets (52% of maps at 5-7*); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 9 chords, ~2.0 beats, 4158 times in 307 sets (49% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```

**motif** - 8 chords, ~1.8 beats, 4645 times in 322 sets (51% of maps at 5-7*); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2
```

**motif** - 9 chords, ~2.0 beats, 3909 times in 297 sets (48% of maps at 5-7*); starts on downbeat 10%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 9 chords, ~2.0 beats, 4028 times in 296 sets (48% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 8 chords, ~1.7 beats, 4367 times in 318 sets (50% of maps at 5-7*); starts on downbeat 13%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 10 chords, ~2.3 beats, 3288 times in 269 sets (45% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

### 5-7*, 16+ chords

**motif** - 16 chords, ~3.8 beats, 1255 times in 137 sets (24% of maps at 5-7*); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2
```

**motif** - 16 chords, ~3.7 beats, 1213 times in 129 sets (22% of maps at 5-7*); starts on downbeat 12%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 16 chords, ~3.7 beats, 1283 times in 137 sets (22% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

**motif** - 17 chords, ~4.0 beats, 1141 times in 118 sets (20% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 17 chords, ~4.0 beats, 1156 times in 125 sets (20% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```

**motif** - 16 chords, ~3.7 beats, 1189 times in 125 sets (20% of maps at 5-7*); starts on downbeat 17%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 17 chords, ~4.0 beats, 1060 times in 118 sets (19% of maps at 5-7*); starts on downbeat 11%
```
1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

**motif** - 18 chords, ~4.3 beats, 1006 times in 112 sets (18% of maps at 5-7*); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

### 7*+, 4-7 chords

**motif** - 7 chords, ~1.5 beats, 7142 times in 379 sets (56% of maps at 7*+); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 6 chords, ~1.3 beats, 8359 times in 414 sets (58% of maps at 7*+); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 6 chords, ~1.2 beats, 7647 times in 396 sets (56% of maps at 7*+); starts on downbeat 10%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 4 chords, ~0.8 beats, 5195 times in 418 sets (85% of maps at 7*+); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:B3
```

**motif** - 7 chords, ~1.5 beats, 5391 times in 346 sets (51% of maps at 7*+); starts on downbeat 12%
```
1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

**motif** - 5 chords, ~1.0 beats, 9299 times in 434 sets (58% of maps at 7*+); starts on downbeat 9%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

**motif** - 5 chords, ~1.0 beats, 13772 times in 471 sets (72% of maps at 7*+); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 7 chords, ~1.5 beats, 5577 times in 358 sets (48% of maps at 7*+); starts on downbeat 9%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2
```

### 7*+, 8-15 chords

**motif** - 8 chords, ~1.8 beats, 5077 times in 336 sets (51% of maps at 7*+); starts on downbeat 0%
```
1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 9 chords, ~2.0 beats, 4158 times in 307 sets (46% of maps at 7*+); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```

**motif** - 9 chords, ~2.0 beats, 3909 times in 297 sets (45% of maps at 7*+); starts on downbeat 10%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 8 chords, ~1.7 beats, 4367 times in 318 sets (46% of maps at 7*+); starts on downbeat 13%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 8 chords, ~1.7 beats, 4838 times in 329 sets (46% of maps at 7*+); starts on downbeat 10%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 10 chords, ~2.3 beats, 3288 times in 269 sets (41% of maps at 7*+); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 9 chords, ~2.0 beats, 4028 times in 296 sets (43% of maps at 7*+); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 8 chords, ~1.8 beats, 4645 times in 322 sets (43% of maps at 7*+); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2
```

### 7*+, 16+ chords

**motif** - 16 chords, ~3.7 beats, 1283 times in 137 sets (20% of maps at 7*+); starts on downbeat 0%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

**motif** - 16 chords, ~3.7 beats, 1213 times in 129 sets (18% of maps at 7*+); starts on downbeat 12%
```
1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 17 chords, ~4.0 beats, 1060 times in 118 sets (16% of maps at 7*+); starts on downbeat 11%
```
1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2
```

**motif** - 18 chords, ~4.3 beats, 1006 times in 112 sets (15% of maps at 7*+); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2
```

**motif** - 17 chords, ~4.0 beats, 1141 times in 118 sets (15% of maps at 7*+); starts on downbeat 0%
```
1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1
```

**motif** - 16 chords, ~3.8 beats, 1255 times in 137 sets (15% of maps at 7*+); starts on downbeat 0%
```
1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2
```

**motif** - 17 chords, ~4.0 beats, 63 times in 24 sets (11% of maps at 7*+); starts on downbeat 3%
```
1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:B2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1
```

**motif** - 16 chords, ~3.7 beats, 65 times in 26 sets (11% of maps at 7*+); starts on downbeat 18%
```
1/4:R2 1/4:L1 1/4:B2 1/4:R1 1/4:B2 1/4:L1 1/4:B2 1/4:R1 1/4:L2 1/4:R1 1/4:B2 1/4:L1 1/4:R2 1/4:L1 1/4:B2 1/4:R1
```


## Level `flow`

### 1-3*, 4-7 chords

**steady** - 4 chords, ~0.9 beats, 11021 times in 524 sets (72% of maps at 1-3*); starts on downbeat 13%
```
slow+x1 accel+x1 steadyx1 steadyx1
```

**steady** - 5 chords, ~1.3 beats, 6102 times in 489 sets (58% of maps at 1-3*); starts on downbeat 16%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1
```

**steady** - 4 chords, ~1.4 beats, 8772 times in 483 sets (63% of maps at 1-3*); starts on downbeat 11%
```
steadyx1 slow+x1 accel+x1 steadyx1
```

**steady** - 4 chords, ~1.1 beats, 3442 times in 364 sets (44% of maps at 1-3*); starts on downbeat 3%
```
steadyx1 accel+x1 steadyx2 slow+x1
```

**steady** - 4 chords, ~0.8 beats, 11117 times in 521 sets (56% of maps at 1-3*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx2
```

**steady** - 4 chords, ~1.2 beats, 5098 times in 447 sets (56% of maps at 1-3*); starts on downbeat 11%
```
steadyx1 accel+x1 steadyx1 slow+x1
```

**steady** - 4 chords, ~0.8 beats, 23799 times in 563 sets (82% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1
```

**speeding up** - 4 chords, ~1.3 beats, 6074 times in 444 sets (54% of maps at 1-3*); starts on downbeat 2%
```
accel+x1 steadyx1 slow+x1 accel+x1
```

### 1-3*, 8-15 chords

**steady** - 11 chords, ~3.0 beats, 3770 times in 440 sets (38% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 10 chords, ~2.7 beats, 4011 times in 449 sets (39% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 9 chords, ~2.3 beats, 4553 times in 464 sets (41% of maps at 1-3*); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.0 beats, 5713 times in 476 sets (43% of maps at 1-3*); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 13 chords, ~3.7 beats, 2619 times in 389 sets (33% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 12 chords, ~3.4 beats, 2926 times in 407 sets (33% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 14 chords, ~4.0 beats, 2435 times in 378 sets (30% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.2 beats, 1866 times in 356 sets (26% of maps at 1-3*); starts on downbeat 27%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

### 1-3*, 16+ chords

**steady** - 16 chords, ~4.6 beats, 1477 times in 306 sets (21% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 21 chords, ~6.4 beats, 1034 times in 256 sets (18% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 23 chords, ~7.0 beats, 983 times in 244 sets (17% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 22 chords, ~6.7 beats, 1004 times in 249 sets (17% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 17 chords, ~4.9 beats, 1291 times in 289 sets (19% of maps at 1-3*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 20 chords, ~6.0 beats, 1092 times in 264 sets (18% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 19 chords, ~5.6 beats, 1149 times in 271 sets (18% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 18 chords, ~5.3 beats, 1192 times in 276 sets (18% of maps at 1-3*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

### 3-5*, 4-7 chords

**steady** - 4 chords, ~0.8 beats, 11117 times in 521 sets (85% of maps at 3-5*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx2
```

**steady** - 4 chords, ~1.2 beats, 5081 times in 439 sets (62% of maps at 3-5*); starts on downbeat 0%
```
steadyx1 steadyx2 slow+x1 accel+x1
```

**steady** - 5 chords, ~1.5 beats, 3778 times in 410 sets (56% of maps at 3-5*); starts on downbeat 3%
```
steadyx1 steadyx1 steadyx2 slow+x1 accel+x1
```

**steady** - 4 chords, ~0.7 beats, 8366 times in 504 sets (80% of maps at 3-5*); starts on downbeat 40%
```
steadyx2 accel+x1 steadyx1 steadyx1
```

**steady** - 4 chords, ~1.0 beats, 9104 times in 501 sets (79% of maps at 3-5*); starts on downbeat 2%
```
steadyx1 steadyx1 steadyx2 slow+x1
```

**steady** - 5 chords, ~1.2 beats, 3907 times in 398 sets (53% of maps at 3-5*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx2 slow+x1
```

**steady** - 4 chords, ~0.9 beats, 11021 times in 524 sets (77% of maps at 3-5*); starts on downbeat 13%
```
slow+x1 accel+x1 steadyx1 steadyx1
```

**steady** - 4 chords, ~1.2 beats, 3903 times in 417 sets (58% of maps at 3-5*); starts on downbeat 33%
```
steadyx2 slow+x1 accel+x1 steadyx1
```

### 3-5*, 8-15 chords

**steady** - 11 chords, ~3.0 beats, 3770 times in 440 sets (62% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 10 chords, ~2.7 beats, 4011 times in 449 sets (63% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.0 beats, 5713 times in 476 sets (70% of maps at 3-5*); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 9 chords, ~2.3 beats, 4553 times in 464 sets (66% of maps at 3-5*); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.2 beats, 1866 times in 356 sets (45% of maps at 3-5*); starts on downbeat 27%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 12 chords, ~3.4 beats, 2926 times in 407 sets (55% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.4 beats, 24425 times in 423 sets (67% of maps at 3-5*); starts on downbeat 15%
```
steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1
```

**steady** - 13 chords, ~3.7 beats, 2619 times in 389 sets (53% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

### 3-5*, 16+ chords

**steady** - 16 chords, ~4.6 beats, 1477 times in 306 sets (38% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 16 chords, ~5.0 beats, 1330 times in 263 sets (38% of maps at 3-5*); starts on downbeat 2%
```
steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx2
```

**steady** - 16 chords, ~5.0 beats, 10702 times in 260 sets (37% of maps at 3-5*); starts on downbeat 16%
```
steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1
```

**steady** - 16 chords, ~4.9 beats, 10552 times in 264 sets (37% of maps at 3-5*); starts on downbeat 1%
```
steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2
```

**steady** - 17 chords, ~4.9 beats, 1291 times in 289 sets (36% of maps at 3-5*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 19 chords, ~5.6 beats, 1149 times in 271 sets (33% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 23 chords, ~7.0 beats, 983 times in 244 sets (30% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 18 chords, ~5.3 beats, 1192 times in 276 sets (33% of maps at 3-5*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

### 5-7*, 4-7 chords

**steady** - 4 chords, ~0.8 beats, 11117 times in 521 sets (77% of maps at 5-7*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx2
```

**steady** - 4 chords, ~0.9 beats, 11021 times in 524 sets (75% of maps at 5-7*); starts on downbeat 13%
```
slow+x1 accel+x1 steadyx1 steadyx1
```

**steady** - 7 chords, ~1.7 beats, 8648 times in 521 sets (84% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 5 chords, ~1.3 beats, 6102 times in 489 sets (66% of maps at 5-7*); starts on downbeat 16%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1
```

**steady** - 4 chords, ~1.2 beats, 5081 times in 439 sets (55% of maps at 5-7*); starts on downbeat 0%
```
steadyx1 steadyx2 slow+x1 accel+x1
```

**steady** - 6 chords, ~1.6 beats, 4072 times in 458 sets (59% of maps at 5-7*); starts on downbeat 19%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 5 chords, ~1.2 beats, 3907 times in 398 sets (48% of maps at 5-7*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx2 slow+x1
```

**steady** - 4 chords, ~0.7 beats, 8366 times in 504 sets (71% of maps at 5-7*); starts on downbeat 40%
```
steadyx2 accel+x1 steadyx1 steadyx1
```

### 5-7*, 8-15 chords

**steady** - 11 chords, ~3.0 beats, 3770 times in 440 sets (64% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~1.7 beats, 1625 times in 350 sets (50% of maps at 5-7*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx2
```

**steady** - 9 chords, ~2.3 beats, 4553 times in 464 sets (71% of maps at 5-7*); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 10 chords, ~2.7 beats, 4011 times in 449 sets (66% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 15 chords, ~4.2 beats, 2238 times in 364 sets (54% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 12 chords, ~3.4 beats, 2926 times in 407 sets (60% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 14 chords, ~4.0 beats, 2435 times in 378 sets (55% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.0 beats, 5713 times in 476 sets (72% of maps at 5-7*); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

### 5-7*, 16+ chords

**steady** - 16 chords, ~4.6 beats, 1477 times in 306 sets (43% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 17 chords, ~4.9 beats, 1291 times in 289 sets (40% of maps at 5-7*); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 20 chords, ~6.0 beats, 1092 times in 264 sets (37% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 21 chords, ~6.4 beats, 1034 times in 256 sets (35% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 19 chords, ~5.6 beats, 1149 times in 271 sets (37% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 23 chords, ~7.0 beats, 983 times in 244 sets (34% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 18 chords, ~5.3 beats, 1192 times in 276 sets (38% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 22 chords, ~6.7 beats, 1004 times in 249 sets (34% of maps at 5-7*); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

### 7*+, 4-7 chords

**steady** - 4 chords, ~0.7 beats, 8366 times in 504 sets (77% of maps at 7*+); starts on downbeat 40%
```
steadyx2 accel+x1 steadyx1 steadyx1
```

**steady** - 4 chords, ~0.8 beats, 11117 times in 521 sets (77% of maps at 7*+); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx2
```

**steady** - 5 chords, ~1.3 beats, 6102 times in 489 sets (69% of maps at 7*+); starts on downbeat 16%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1
```

**steady** - 4 chords, ~1.1 beats, 5860 times in 478 sets (75% of maps at 7*+); starts on downbeat 2%
```
steadyx1 steadyx1 steadyx2 slow+x2
```

**steady** - 4 chords, ~0.9 beats, 11021 times in 524 sets (75% of maps at 7*+); starts on downbeat 13%
```
slow+x1 accel+x1 steadyx1 steadyx1
```

**steady** - 5 chords, ~1.2 beats, 2171 times in 375 sets (67% of maps at 7*+); starts on downbeat 14%
```
steadyx2 steadyx1 accel+x1 steadyx1 steadyx1
```

**steady** - 4 chords, ~1.1 beats, 3743 times in 443 sets (74% of maps at 7*+); starts on downbeat 19%
```
steadyx2 steadyx1 accel+x1 steadyx1
```

**steady** - 5 chords, ~1.2 beats, 2209 times in 350 sets (49% of maps at 7*+); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx2 slow+x2
```

### 7*+, 8-15 chords

**steady** - 8 chords, ~1.7 beats, 1625 times in 350 sets (51% of maps at 7*+); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx2
```

**steady** - 8 chords, ~2.0 beats, 5713 times in 476 sets (69% of maps at 7*+); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.1 beats, 6182 times in 450 sets (67% of maps at 7*+); starts on downbeat 1%
```
steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx2
```

**steady** - 8 chords, ~2.2 beats, 1866 times in 356 sets (44% of maps at 7*+); starts on downbeat 27%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~1.8 beats, 2908 times in 398 sets (65% of maps at 7*+); starts on downbeat 38%
```
steadyx1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 8 chords, ~2.4 beats, 23841 times in 422 sets (64% of maps at 7*+); starts on downbeat 1%
```
steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2 steadyx1 steadyx2
```

**steady** - 15 chords, ~4.2 beats, 2238 times in 364 sets (46% of maps at 7*+); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 9 chords, ~2.3 beats, 4553 times in 464 sets (59% of maps at 7*+); starts on downbeat 2%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

### 7*+, 16+ chords

**steady** - 17 chords, ~4.9 beats, 1291 times in 289 sets (33% of maps at 7*+); starts on downbeat 0%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 16 chords, ~4.6 beats, 1477 times in 306 sets (33% of maps at 7*+); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 19 chords, ~6.1 beats, 239 times in 111 sets (20% of maps at 7*+); starts on downbeat 40%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 18 chords, ~5.3 beats, 1192 times in 276 sets (30% of maps at 7*+); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 18 chords, ~5.7 beats, 254 times in 117 sets (20% of maps at 7*+); starts on downbeat 39%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 19 chords, ~5.6 beats, 1149 times in 271 sets (28% of maps at 7*+); starts on downbeat 1%
```
accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 17 chords, ~5.3 beats, 281 times in 127 sets (20% of maps at 7*+); starts on downbeat 41%
```
slow+x1 accel+x1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1 steadyx1
```

**steady** - 18 chords, ~4.9 beats, 5970 times in 246 sets (28% of maps at 7*+); starts on downbeat 3%
```
steadyx1 steadyx1 steadyx2 steadyx1 steadyx1 steadyx1 steadyx2 steadyx1 steadyx1 steadyx1 steadyx2 steadyx1 steadyx1 steadyx1 steadyx2 steadyx1 steadyx1 steadyx1
```


## Level `ln`

### 1-3*, 4-7 chords

**stream** - 4 chords, ~1.0 beats, 19778 times in 541 sets (83% of maps at 1-3*); starts on downbeat 7%
```
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stairs** - 4 chords, ~0.8 beats, 24889 times in 549 sets (82% of maps at 1-3*); starts on downbeat 11%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 4 chords, ~1.0 beats, 16285 times in 534 sets (79% of maps at 1-3*); starts on downbeat 6%
```
|.#..| 
|..#.| 
|#...| 
|...#| 
```

**stream** - 4 chords, ~1.0 beats, 15815 times in 530 sets (79% of maps at 1-3*); starts on downbeat 5%
```
|...#| 
|#...| 
|..#.| 
|.#..| 
```

**stream** - 4 chords, ~0.8 beats, 17401 times in 539 sets (79% of maps at 1-3*); starts on downbeat 4%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
```

**stream** - 4 chords, ~0.8 beats, 18589 times in 541 sets (77% of maps at 1-3*); starts on downbeat 4%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 4 chords, ~0.9 beats, 16000 times in 532 sets (76% of maps at 1-3*); starts on downbeat 8%
```
|..#.| 
|...#| 
|.#..| 
|#...| 
```

**stream** - 4 chords, ~1.0 beats, 15345 times in 520 sets (75% of maps at 1-3*); starts on downbeat 5%
```
|..#.| 
|#...| 
|...#| 
|.#..| 
```

### 1-3*, 8-15 chords

**stream** - 8 chords, ~2.7 beats, 1178 times in 243 sets (24% of maps at 1-3*); starts on downbeat 13%
```
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stream** - 8 chords, ~2.5 beats, 1090 times in 217 sets (24% of maps at 1-3*); starts on downbeat 9%
```
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
```

**stream** - 8 chords, ~2.6 beats, 1176 times in 233 sets (24% of maps at 1-3*); starts on downbeat 11%
```
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
```

**stream** - 8 chords, ~2.6 beats, 1099 times in 217 sets (22% of maps at 1-3*); starts on downbeat 10%
```
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
```

**stream** - 8 chords, ~2.7 beats, 531 times in 195 sets (20% of maps at 1-3*); starts on downbeat 5%
```
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 9 chords, ~3.1 beats, 707 times in 169 sets (17% of maps at 1-3*); starts on downbeat 11%
```
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
```

**stream** - 8 chords, ~2.3 beats, 770 times in 186 sets (18% of maps at 1-3*); starts on downbeat 3%
```
|#...| 
|..#.| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
```

**stream** - 8 chords, ~2.4 beats, 601 times in 203 sets (17% of maps at 1-3*); starts on downbeat 12%
```
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

### 1-3*, 16+ chords

**stream** - 16 chords, ~6.7 beats, 24 times in 16 sets (2% of maps at 1-3*); starts on downbeat 21%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|..#.| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|.#..| 
|#...| 
|..#.| 
|.#..| 
```

**stream** - 16 chords, ~5.2 beats, 94 times in 30 sets (1% of maps at 1-3*); starts on downbeat 23%
```
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stream** - 16 chords, ~4.9 beats, 97 times in 31 sets (1% of maps at 1-3*); starts on downbeat 16%
```
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
|.#..| 
|..#.| 
|#...| 
|...#| 
```

**jumpstream** - 17 chords, ~8.7 beats, 5 times in 4 sets (1% of maps at 1-3*); starts on downbeat 0%
```
|.#..| 
|..##| 
|.#..| 
|#.#.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.#.#| 
|..#.| 
|##..| 
|...#| 
|#.#.| 
|.#..| 
```

**jumpstream** - 17 chords, ~5.8 beats, 11 times in 6 sets (1% of maps at 1-3*); starts on downbeat 0%
```
|..#.| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.#.#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|..#.| 
|##..| 
|..#.| 
```

**jumpstream** - 16 chords, ~15.0 beats, 3 times in 3 sets (1% of maps at 1-3*); starts on downbeat 33%
```
|..#.| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|##..| 
|...#| 
|.##.| 
|#...| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|..##| 
|.#..| 
|#.#.| 
```

**jumpstream** - 16 chords, ~4.4 beats, 31 times in 11 sets (1% of maps at 1-3*); starts on downbeat 10%
```
|..#.| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.#.#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|..#.| 
|##..| 
```

**jumpstream** - 16 chords, ~9.5 beats, 4 times in 3 sets (1% of maps at 1-3*); starts on downbeat 50%
```
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|#.#.| 
|.#..| 
|..##| 
|#...| 
|.#.#| 
|..#.| 
|##..| 
|...#| 
|#.#.| 
|.#..| 
|#..#| 
```

### 3-5*, 4-7 chords

**stairs** - 4 chords, ~0.8 beats, 24889 times in 549 sets (86% of maps at 3-5*); starts on downbeat 11%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 4 chords, ~0.8 beats, 18589 times in 541 sets (84% of maps at 3-5*); starts on downbeat 4%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 4 chords, ~0.8 beats, 17401 times in 539 sets (82% of maps at 3-5*); starts on downbeat 4%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
```

**stream** - 4 chords, ~0.9 beats, 16000 times in 532 sets (82% of maps at 3-5*); starts on downbeat 8%
```
|..#.| 
|...#| 
|.#..| 
|#...| 
```

**stream** - 5 chords, ~1.0 beats, 9993 times in 484 sets (73% of maps at 3-5*); starts on downbeat 12%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 5 chords, ~1.0 beats, 10659 times in 499 sets (72% of maps at 3-5*); starts on downbeat 3%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 4 chords, ~0.8 beats, 16144 times in 534 sets (81% of maps at 3-5*); starts on downbeat 3%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
```

**jumpstream** - 5 chords, ~1.3 beats, 4054 times in 371 sets (58% of maps at 3-5*); starts on downbeat 18%
```
|##..| 
|..#.| 
|#..#| 
|.#..| 
|..##| 
```

### 3-5*, 8-15 chords

**jumpstream** - 8 chords, ~2.2 beats, 577 times in 144 sets (21% of maps at 3-5*); starts on downbeat 2%
```
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|..#.| 
```

**jumpstream** - 8 chords, ~2.2 beats, 568 times in 137 sets (20% of maps at 3-5*); starts on downbeat 21%
```
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|..#.| 
|##..| 
```

**stream** - 8 chords, ~1.5 beats, 2984 times in 280 sets (38% of maps at 3-5*); starts on downbeat 16%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**jumpstream** - 8 chords, ~2.2 beats, 637 times in 148 sets (19% of maps at 3-5*); starts on downbeat 1%
```
|#..#| 
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
```

**jumpstream** - 8 chords, ~2.2 beats, 621 times in 153 sets (18% of maps at 3-5*); starts on downbeat 17%
```
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
```

**jumpstream** - 8 chords, ~2.1 beats, 615 times in 145 sets (18% of maps at 3-5*); starts on downbeat 19%
```
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
```

**jumpstream** - 9 chords, ~2.4 beats, 436 times in 110 sets (16% of maps at 3-5*); starts on downbeat 19%
```
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|..#.| 
|##..| 
```

**jumpstream** - 9 chords, ~2.5 beats, 502 times in 128 sets (16% of maps at 3-5*); starts on downbeat 1%
```
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
```

### 3-5*, 16+ chords

**stream** - 16 chords, ~2.8 beats, 586 times in 85 sets (10% of maps at 3-5*); starts on downbeat 25%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 16 chords, ~2.9 beats, 488 times in 75 sets (9% of maps at 3-5*); starts on downbeat 3%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
```

**stream** - 16 chords, ~2.9 beats, 491 times in 69 sets (8% of maps at 3-5*); starts on downbeat 2%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 17 chords, ~3.0 beats, 459 times in 65 sets (8% of maps at 3-5*); starts on downbeat 1%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 16 chords, ~2.9 beats, 442 times in 67 sets (8% of maps at 3-5*); starts on downbeat 2%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**stream** - 17 chords, ~3.0 beats, 451 times in 64 sets (7% of maps at 3-5*); starts on downbeat 22%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 17 chords, ~3.0 beats, 414 times in 62 sets (7% of maps at 3-5*); starts on downbeat 1%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**stream** - 18 chords, ~3.1 beats, 423 times in 60 sets (7% of maps at 3-5*); starts on downbeat 21%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

### 5-7*, 4-7 chords

**stairs** - 4 chords, ~0.8 beats, 24889 times in 549 sets (84% of maps at 5-7*); starts on downbeat 11%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**jumpstream** - 5 chords, ~0.8 beats, 3591 times in 419 sets (60% of maps at 5-7*); starts on downbeat 5%
```
|##..| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 4 chords, ~0.8 beats, 18589 times in 541 sets (83% of maps at 5-7*); starts on downbeat 4%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 5 chords, ~1.0 beats, 10659 times in 499 sets (74% of maps at 5-7*); starts on downbeat 3%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 4 chords, ~0.8 beats, 16144 times in 534 sets (80% of maps at 5-7*); starts on downbeat 3%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
```

**stream** - 4 chords, ~1.0 beats, 19778 times in 541 sets (79% of maps at 5-7*); starts on downbeat 7%
```
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stream** - 4 chords, ~0.8 beats, 17401 times in 539 sets (79% of maps at 5-7*); starts on downbeat 4%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
```

**stream** - 5 chords, ~1.0 beats, 9313 times in 489 sets (71% of maps at 5-7*); starts on downbeat 3%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
```

### 5-7*, 8-15 chords

**stream** - 8 chords, ~1.5 beats, 2984 times in 280 sets (40% of maps at 5-7*); starts on downbeat 16%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**jumpstream** - 8 chords, ~2.2 beats, 637 times in 148 sets (19% of maps at 5-7*); starts on downbeat 1%
```
|#..#| 
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
```

**stream** - 8 chords, ~1.4 beats, 2379 times in 247 sets (37% of maps at 5-7*); starts on downbeat 3%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
```

**jumpstream** - 8 chords, ~2.2 beats, 621 times in 153 sets (18% of maps at 5-7*); starts on downbeat 17%
```
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
```

**jumpstream** - 8 chords, ~2.3 beats, 545 times in 132 sets (18% of maps at 5-7*); starts on downbeat 1%
```
|..##| 
|.#..| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
```

**stream** - 8 chords, ~1.5 beats, 2314 times in 244 sets (35% of maps at 5-7*); starts on downbeat 2%
```
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

**stream** - 8 chords, ~1.5 beats, 2162 times in 243 sets (34% of maps at 5-7*); starts on downbeat 3%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**light jumpstream** - 8 chords, ~1.2 beats, 596 times in 171 sets (27% of maps at 5-7*); starts on downbeat 1%
```
|..##| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

### 5-7*, 16+ chords

**stream** - 16 chords, ~2.8 beats, 586 times in 85 sets (15% of maps at 5-7*); starts on downbeat 25%
```
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
```

**stream** - 16 chords, ~2.9 beats, 488 times in 75 sets (12% of maps at 5-7*); starts on downbeat 3%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
```

**stream** - 17 chords, ~3.0 beats, 451 times in 64 sets (11% of maps at 5-7*); starts on downbeat 22%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 16 chords, ~2.9 beats, 491 times in 69 sets (11% of maps at 5-7*); starts on downbeat 2%
```
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 18 chords, ~3.1 beats, 423 times in 60 sets (10% of maps at 5-7*); starts on downbeat 21%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**light jumpstream** - 16 chords, ~2.4 beats, 68 times in 37 sets (9% of maps at 5-7*); starts on downbeat 16%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|..##| 
```

**stream** - 16 chords, ~2.9 beats, 442 times in 67 sets (11% of maps at 5-7*); starts on downbeat 2%
```
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
```

**stream** - 17 chords, ~3.0 beats, 459 times in 65 sets (10% of maps at 5-7*); starts on downbeat 1%
```
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
|#...| 
|...#| 
|..#.| 
|.#..| 
```

### 7*+, 4-7 chords

**stream** - 4 chords, ~1.0 beats, 16285 times in 534 sets (87% of maps at 7*+); starts on downbeat 6%
```
|.#..| 
|..#.| 
|#...| 
|...#| 
```

**stream** - 5 chords, ~1.3 beats, 7317 times in 489 sets (75% of maps at 7*+); starts on downbeat 6%
```
|.#..| 
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stairs** - 4 chords, ~0.8 beats, 24889 times in 549 sets (82% of maps at 7*+); starts on downbeat 11%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
```

**stream** - 5 chords, ~1.2 beats, 3982 times in 456 sets (72% of maps at 7*+); starts on downbeat 4%
```
|...#| 
|..#.| 
|#...| 
|.#..| 
|..#.| 
```

**stream** - 4 chords, ~1.0 beats, 19778 times in 541 sets (80% of maps at 7*+); starts on downbeat 7%
```
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stream** - 4 chords, ~0.8 beats, 17401 times in 539 sets (80% of maps at 7*+); starts on downbeat 4%
```
|.#..| 
|..#.| 
|...#| 
|#...| 
```

**jumpstream** - 4 chords, ~0.8 beats, 6895 times in 492 sets (78% of maps at 7*+); starts on downbeat 28%
```
|..#.| 
|.#..| 
|#...| 
|..##| 
```

**stream** - 4 chords, ~1.0 beats, 15815 times in 530 sets (75% of maps at 7*+); starts on downbeat 5%
```
|...#| 
|#...| 
|..#.| 
|.#..| 
```

### 7*+, 8-15 chords

**chordjack mix** - 11 chords, ~2.5 beats, 23 times in 3 sets (19% of maps at 7*+); starts on downbeat 0%
```
|####| 
|##..| 
|####| 
|..##| 
|####| 
|#..#| 
|####| 
|##..| 
|####| 
|.##.| 
|####| 
```

**stream** - 8 chords, ~1.4 beats, 2379 times in 247 sets (28% of maps at 7*+); starts on downbeat 3%
```
|..#.| 
|...#| 
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|.#..| 
```

**chordjack mix** - 9 chords, ~2.5 beats, 44 times in 10 sets (21% of maps at 7*+); starts on downbeat 5%
```
|####| 
|#.#.| 
|####| 
|.#.#| 
|####| 
|##..| 
|####| 
|..##| 
|####| 
```

**chord jacks** - 8 chords, ~2.2 beats, 45 times in 11 sets (21% of maps at 7*+); starts on downbeat 0%
```
|####| 
|#.#.| 
|####| 
|.#.#| 
|####| 
|##..| 
|####| 
|..##| 
```

**chordjack mix** - 8 chords, ~2.1 beats, 50 times in 10 sets (21% of maps at 7*+); starts on downbeat 4%
```
|.#.#| 
|####| 
|#.#.| 
|####| 
|..##| 
|####| 
|##..| 
|####| 
```

**stream** - 8 chords, ~2.2 beats, 574 times in 196 sets (26% of maps at 7*+); starts on downbeat 8%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|...#| 
```

**stream** - 8 chords, ~1.8 beats, 446 times in 160 sets (26% of maps at 7*+); starts on downbeat 10%
```
|#...| 
|.#..| 
|..#.| 
|...#| 
|#...| 
|..#.| 
|.#..| 
|#...| 
```

**chordjack mix** - 9 chords, ~2.0 beats, 44 times in 4 sets (19% of maps at 7*+); starts on downbeat 11%
```
|####| 
|..##| 
|####| 
|##..| 
|####| 
|.##.| 
|####| 
|#.#.| 
|####| 
```

### 7*+, 16+ chords

**jumpstream** - 18 chords, ~5.3 beats, 8 times in 3 sets (8% of maps at 7*+); starts on downbeat 0%
```
|.###| 
|#...| 
|##.#| 
|..#.| 
|#.##| 
|.#..| 
|###.| 
|...#| 
|#.##| 
|.#..| 
|##.#| 
|..#.| 
|.###| 
|#...| 
|##.#| 
|..#.| 
|#.##| 
|.#..| 
```

**chordjack mix** - 17 chords, ~5.6 beats, 10 times in 4 sets (8% of maps at 7*+); starts on downbeat 80%
```
|##.#| 
|..##| 
|###.| 
|#..#| 
|.###| 
|##..| 
|#.##| 
|.##.| 
|##.#| 
|..##| 
|###.| 
|#..#| 
|.###| 
|##..| 
|#.##| 
|.##.| 
|##.#| 
```

**jumpstream** - 16 chords, ~5.0 beats, 9 times in 4 sets (8% of maps at 7*+); starts on downbeat 0%
```
|.###| 
|#...| 
|##.#| 
|..#.| 
|#.##| 
|.#..| 
|###.| 
|...#| 
|#.##| 
|.#..| 
|##.#| 
|..#.| 
|.###| 
|#...| 
|##.#| 
|..#.| 
```

**jumpstream** - 19 chords, ~5.5 beats, 18 times in 6 sets (3% of maps at 7*+); starts on downbeat 0%
```
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
```

**jumpstream** - 18 chords, ~5.1 beats, 19 times in 7 sets (3% of maps at 7*+); starts on downbeat 0%
```
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.##.| 
```

**jumpstream** - 17 chords, ~4.7 beats, 27 times in 11 sets (3% of maps at 7*+); starts on downbeat 0%
```
|...#| 
|.##.| 
|#...| 
|..##| 
|.#..| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|..#.| 
|##..| 
|...#| 
```

**jumpstream** - 17 chords, ~4.8 beats, 20 times in 8 sets (3% of maps at 7*+); starts on downbeat 0%
```
|#...| 
|..##| 
|.#..| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
|#...| 
```

**jumpstream** - 16 chords, ~4.5 beats, 21 times in 9 sets (3% of maps at 7*+); starts on downbeat 5%
```
|#...| 
|..##| 
|.#..| 
|#..#| 
|.#..| 
|..##| 
|#...| 
|.##.| 
|...#| 
|##..| 
|..#.| 
|#..#| 
|..#.| 
|##..| 
|...#| 
|.##.| 
```
