# Composer vs human, zone by zone

Break periods ([Events] 2,start,end): human difficulties paired 3, composer 0.

885 zones of 9 songs, 41 composed difficulties.

Distance per statistic (0 = identical; mean over zones, and the mean of the worst / best fifth):

| statistic | all | worst 20 % | best 20 % |
|---|---|---|---|
| nps | 0.446 | 1.094 | 0.200 |
| chord | 0.104 | 0.240 | 0.038 |
| jack | 0.120 | 0.211 | 0.041 |
| ln | 0.181 | 0.244 | 0.104 |
| rhythm | 0.378 | 0.656 | 0.164 |
| lanes | 0.532 | 0.715 | 0.383 |
| shape | 0.865 | 0.655 | 0.876 |
| motifs | 0.984 | 0.996 | 0.972 |
| variety | -0.038 | 0.053 | -0.047 |
| placement | 0.374 | 0.590 | 0.252 |
| rest | 0.014 | 0.068 | 0.000 |
| on_human | 0.722 | 0.479 | 0.831 |
| human_hit | 0.530 | 0.341 | 0.665 |

Largest contributions to the worst zones: nps 1.09, rhythm 0.98, lanes 0.71, motifs 0.70, shape 0.66

## Systematic differences (mean over zones, human vs composer)

| measure | human | composer |
|---|---|---|
| notes per second | 10.293 | 7.977 |
| jack share | 0.158 | 0.058 |
| LN share | 0.194 | 0.070 |
| chord share (2+ notes) | 0.286 | 0.377 |
| distinct 4-chord shapes / chord | 0.825 | 0.863 |
| chords copied from earlier (exact+mirror) | 0.139 | 0.017 |
| chords in a variation copy | 0.039 | 0.021 |
| off-grid chords | 0.022 | 0.011 |
| library lane motifs per zone | 158.800 | 125.158 |
| notes in human rests (break / silence) | 0.000 | 0.102 |
| notes after the song's last human note | 0.000 | 0.139 |

Rhythm classes over all zones (share of chords):

| gap | human | composer |
|---|---|---|
| 1/4 | 41.9% | 28.4% |
| 1/2 | 27.9% | 45.4% |
| 1 | 5.9% | 13.9% |
| 1/8 | 5.4% | 5.1% |
| 1/6 | 7.1% | 0.9% |
| 1/3 | 5.2% | 0.5% |
| off | 1.8% | 1.8% |
| 2 | 0.6% | 1.6% |
| 3/4 | 1.4% | 0.3% |
| 1/12 | 1.1% | 0.2% |
| 3/2 | 0.4% | 0.7% |
| 3 | 0.1% | 0.5% |
| 3/8 | 0.4% | 0.1% |
| rest | 0.1% | 0.4% |
| 2/3 | 0.3% | 0.0% |
| 4 | 0.2% | 0.2% |
| 3/16 | 0.2% | 0.0% |
| 1/16 | 0.0% | 0.1% |

## Per song and difficulty

| song | human diff | composer diff | zones | mean distance | worst zone |
|---|---|---|---|---|---|
| 1964340 | GET DOWN [0.85] (6.06*) | Composer 6.06* (6.05*) | 11 | 5.24 | 0.0-7.4s (8.25) |
| 631020 | this isn't rocket science (5.38*) | Composer 5.38* (5.38*) | 49 | 5.15 | 228.6-243.1s (6.22) |
| 1755294 | The Sorcerer's Apprentice (5.97*) | Composer 5.97* (5.94*) | 31 | 5.08 | 264.6-269.3s (9.56) |
| 1755294 | Extra (5.22*) | Composer 5.22* (5.24*) | 31 | 4.71 | 264.6-269.3s (9.60) |
| 2362621 | MrMoai's Lava Chicken (5.67*) | Composer 5.67* (5.64*) | 10 | 4.49 | 88.2-89.7s (8.85) |
| 1125388 | mafia (extra) (5.20*) | Composer 5.20* (5.17*) | 21 | 4.45 | 249.6-256.5s (7.60) |
| 2400881 | XingRen's Extra (4.37*) | Composer 4.37* (4.37*) | 16 | 4.40 | 0.0-8.7s (7.60) |
| 1579707 | Easy (1.93*) | Composer 1.93* (1.91*) | 39 | 4.31 | 6.2-12.4s (7.78) |
| 360565 | Disappearance (5.11*) | Composer 5.11* (5.12*) | 27 | 4.30 | 11.4-22.6s (7.61) |
| 1755294 | Another (4.46*) | Composer 4.46* (4.47*) | 31 | 4.18 | 264.6-269.3s (9.59) |
| 2400881 | O! S! U! (5.50*) | Composer 5.50* (5.48*) | 16 | 4.15 | 0.0-8.7s (8.03) |
| 1755294 | Insane (3.59*) | Composer 3.59* (3.59*) | 31 | 4.14 | 264.6-269.3s (9.03) |
| 1125388 | capo (insane) (4.68*) | Composer 4.68* (4.64*) | 21 | 4.14 | 249.6-256.5s (7.64) |
| 1755294 | Normal (2.27*) | Composer 2.27* (2.30*) | 31 | 4.11 | 264.6-269.3s (8.85) |
| 1755294 | Hard (2.95*) | Composer 2.95* (2.94*) | 31 | 4.11 | 264.6-269.3s (9.21) |
| 1579707 | Q U A T T U O R (5.80*) | Composer 5.80* (5.81*) | 39 | 4.11 | 0.0-6.2s (8.44) |
| 1579707 | Normal (2.43*) | Composer 2.43* (2.44*) | 39 | 4.07 | 6.2-12.4s (7.62) |
| 360565 | HD (2.69*) | Composer 2.69* (2.69*) | 27 | 3.99 | 11.4-22.6s (7.51) |
| 360565 | Hydria's SC (4.44*) | Composer 4.44* (4.47*) | 25 | 3.97 | 139.2-143.2s (5.80) |
| 2400881 | Easy (1.42*) | Composer 1.42* (1.45*) | 16 | 3.93 | 0.0-8.7s (7.31) |
| 2362621 | Another (4.96*) | Composer 4.96* (4.98*) | 9 | 3.91 | 58.5-66.5s (4.94) |
| 872940 | Medium (2.18*) | Composer 2.18* (2.19*) | 9 | 3.89 | 72.1-77.8s (7.62) |
| 2400881 | Expert (4.24*) | Composer 4.24* (4.26*) | 16 | 3.86 | 0.0-8.7s (7.63) |
| 1579707 | K A I S E R (5.21*) | Composer 5.21* (5.21*) | 39 | 3.85 | 0.0-6.2s (8.44) |
| 872940 | Radioactive Noodles (4.94*) | Composer 4.94* (4.96*) | 9 | 3.84 | 94.7-117.3s (5.05) |
| 360565 | NM (2.00*) | Composer 2.00* (2.02*) | 27 | 3.84 | 0.0-11.4s (7.56) |
| 1579707 | Hard (3.44*) | Composer 3.44* (3.44*) | 39 | 3.81 | 230.7-240.0s (7.92) |
| 1579707 | Insane (4.45*) | Composer 4.45* (4.44*) | 39 | 3.78 | 0.0-6.2s (7.80) |
| 2400881 | Insane (3.87*) | Composer 3.87* (3.90*) | 16 | 3.78 | 0.0-8.7s (7.76) |
| 872940 | Normal (1.89*) | Composer 1.89* (1.87*) | 9 | 3.73 | 72.1-77.8s (6.55) |
| 360565 | MX (3.55*) | Composer 3.55* (3.55*) | 27 | 3.72 | 11.4-22.6s (7.51) |
| 2400881 | Hard (3.05*) | Composer 3.05* (3.06*) | 16 | 3.66 | 0.0-8.7s (7.76) |
| 872940 | Insane (4.16*) | Composer 4.16* (4.14*) | 9 | 3.63 | 72.1-77.8s (4.30) |
| 2400881 | Normal (2.10*) | Composer 2.10* (2.10*) | 16 | 3.56 | 0.0-8.7s (7.84) |
| 2362621 | Insane (4.02*) | Composer 4.02* (4.01*) | 9 | 3.43 | 58.5-66.5s (4.65) |
| 2362621 | Menphiss' Easy (1.47*) | Composer 1.47* (1.47*) | 9 | 3.43 | 58.5-66.5s (4.39) |
| 872940 | Ash's Easy (1.51*) | Composer 1.51* (1.51*) | 9 | 3.40 | 72.1-77.8s (6.91) |
| 2362621 | Utiba's Normal (2.71*) | Composer 2.71* (2.73*) | 9 | 3.32 | 0.0-10.5s (4.14) |
| 872940 | Hyper (3.59*) | Composer 3.59* (3.58*) | 9 | 3.31 | 35.4-41.1s (4.08) |
| 2362621 | Enma's Hard (3.02*) | Composer 3.02* (3.01*) | 9 | 3.20 | 58.5-66.5s (4.15) |
| 872940 | Hard (3.04*) | Composer 3.04* (3.03*) | 9 | 2.97 | 72.1-77.8s (4.24) |

## Worst zones

### 1755294: 264.6-269.3s (#30 intensity 0.00), human Extra 5.22* vs composer 5.24*, distance 9.60

- notes/s: human 0.0, composer 3.6; chords 1/2/3/4: human 0/0/0/0, composer 6/4/1/0
- jacks: human 0%, composer 6%; long notes: human 0%, composer 12%
- notes where all human maps rest (break / silence / after the end): composer 17 of 17; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 0.00, composer 1.00
- rhythm (gap to previous chord): human ; composer 1/2 64%, 1 27%, 3/2 9%
- copies of earlier passages: human ; composer 
- lane motifs only the composer uses here: jumpstream [4 12 2 13]; jumpstream [3 4 11 1]; jumpstream [2 13 8 3]; jumpstream [11 1 12 3]

```
human: Extra           composer: Composer 5.22*
                       192,192,264589,1,0,0:0:0:0:
                       64,192,264824,1,0,0:0:0:0:
                       192,192,264824,1,0,0:0:0:0:
                       320,192,264941,1,0,0:0:0:0:
                       64,192,265058,1,0,0:0:0:0:
                       192,192,265058,1,0,0:0:0:0:
                       448,192,265058,1,0,0:0:0:0:
                       64,192,265175,1,0,0:0:0:0:
                       320,192,265292,1,0,0:0:0:0:
                       448,192,265292,1,0,0:0:0:0:
                       64,192,265527,1,0,0:0:0:0:
                       192,192,265527,1,0,0:0:0:0:
                       320,192,265644,1,0,0:0:0:0:
                       64,192,265761,1,0,0:0:0:0:
                       448,192,265761,128,0,266230:0:0:0:0:
                       320,192,266113,1,0,0:0:0:0:
                       192,192,266230,128,0,268105:0:0:0:0:

                    264589 |.#..|
                    264824 |##..|
                    264941 |..#.|
                    265058 |##.#|
                    265175 |#...|
                    265292 |..##|
                    265527 |##..|
                    265644 |..#.|
                    265761 |#..O|
                    266113 |..#.|
                    266230 |.O..|
```

### 1755294: 264.6-269.3s (#30 intensity 0.00), human Another 4.46* vs composer 4.47*, distance 9.59

- notes/s: human 0.0, composer 3.6; chords 1/2/3/4: human 0/0/0/0, composer 6/4/1/0
- jacks: human 0%, composer 6%; long notes: human 0%, composer 12%
- notes where all human maps rest (break / silence / after the end): composer 17 of 17; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 0.00, composer 1.00
- rhythm (gap to previous chord): human ; composer 1/2 64%, 1 27%, 3/2 9%
- copies of earlier passages: human ; composer 
- lane motifs only the composer uses here: jumpstream [4 3 8 7]; jumpstream [3 8 7 4]; jumpstream [8 7 4 10]; jumpstream [14 2 5 12]

```
human: Another         composer: Composer 4.46*
                       192,192,264589,1,0,0:0:0:0:
                       320,192,264824,1,0,0:0:0:0:
                       448,192,264824,1,0,0:0:0:0:
                       64,192,264941,1,0,0:0:0:0:
                       192,192,265058,1,0,0:0:0:0:
                       320,192,265058,1,0,0:0:0:0:
                       448,192,265058,1,0,0:0:0:0:
                       192,192,265175,1,0,0:0:0:0:
                       64,192,265292,1,0,0:0:0:0:
                       320,192,265292,1,0,0:0:0:0:
                       320,192,265527,1,0,0:0:0:0:
                       448,192,265527,1,0,0:0:0:0:
                       192,192,265644,1,0,0:0:0:0:
                       64,192,265761,1,0,0:0:0:0:
                       448,192,265761,128,0,265996:0:0:0:0:
                       320,192,266113,1,0,0:0:0:0:
                       192,192,266230,128,0,268105:0:0:0:0:

                    264589 |.#..|
                    264824 |..##|
                    264941 |#...|
                    265058 |.###|
                    265175 |.#..|
                    265292 |#.#.|
                    265527 |..##|
                    265644 |.#..|
                    265761 |#..O|
                    266113 |..#.|
                    266230 |.O..|
```

### 1755294: 264.6-269.3s (#30 intensity 0.00), human The Sorcerer's Apprentice 5.97* vs composer 5.94*, distance 9.56

- notes/s: human 0.0, composer 3.8; chords 1/2/3/4: human 0/0/0/0, composer 6/3/2/0
- jacks: human 0%, composer 0%; long notes: human 0%, composer 17%
- notes where all human maps rest (break / silence / after the end): composer 18 of 18; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 0.00, composer 1.00
- rhythm (gap to previous chord): human ; composer 1/2 64%, 1 27%, 3/2 9%
- copies of earlier passages: human ; composer 
- lane motifs only the composer uses here: jumpstream [4 12 2 13]; jumpstream [3 4 11 4]; jumpstream [2 13 2 9]; jumpstream [11 4 9 6]

```
human: The Sorcerer'   composer: Composer 5.97*
                       320,192,264589,1,0,0:0:0:0:
                       320,192,264824,1,0,0:0:0:0:
                       448,192,264824,1,0,0:0:0:0:
                       192,192,264941,1,0,0:0:0:0:
                       64,192,265058,1,0,0:0:0:0:
                       320,192,265058,1,0,0:0:0:0:
                       448,192,265058,1,0,0:0:0:0:
                       192,192,265175,1,0,0:0:0:0:
                       64,192,265292,128,0,265468:0:0:0:0:
                       448,192,265292,1,0,0:0:0:0:
                       192,192,265527,1,0,0:0:0:0:
                       320,192,265527,1,0,0:0:0:0:
                       448,192,265644,1,0,0:0:0:0:
                       64,192,265761,128,0,266464:0:0:0:0:
                       192,192,265761,1,0,0:0:0:0:
                       320,192,265761,1,0,0:0:0:0:
                       448,192,266113,1,0,0:0:0:0:
                       192,192,266230,128,0,266933:0:0:0:0:

                    264589 |..#.|
                    264824 |..##|
                    264941 |.#..|
                    265058 |#.##|
                    265175 |.#..|
                    265292 |O..#|
                    265527 |.##.|
                    265644 |...#|
                    265761 |O##.|
                    266113 |...#|
                    266230 |.O..|
```

### 1755294: 264.6-269.3s (#30 intensity 0.00), human Hard 2.95* vs composer 2.94*, distance 9.21

- notes/s: human 0.0, composer 1.5; chords 1/2/3/4: human 0/0/0/0, composer 3/2/0/0
- jacks: human 0%, composer 14%; long notes: human 0%, composer 29%
- notes where all human maps rest (break / silence / after the end): composer 7 of 7; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 0.00, composer 1.00
- rhythm (gap to previous chord): human ; composer 2 40%, 1 40%, 3 20%
- copies of earlier passages: human ; composer 
- lane motifs only the composer uses here: jumpstream [8 3 4 9]; jumpstream [3 4 9 2]

```
human: Hard            composer: Composer 2.95*
                       64,192,264824,1,0,0:0:0:0:
                       320,192,265058,128,0,265527:0:0:0:0:
                       448,192,265058,1,0,0:0:0:0:
                       192,192,265527,1,0,0:0:0:0:
                       64,192,265761,1,0,0:0:0:0:
                       448,192,265761,1,0,0:0:0:0:
                       320,192,266230,128,0,267168:0:0:0:0:

                    264824 |#...|
                    265058 |..O#|
                    265527 |.#..|
                    265761 |#..#|
                    266230 |..O.|
```

### 1755294: 264.6-269.3s (#30 intensity 0.00), human Insane 3.59* vs composer 3.59*, distance 9.03

- notes/s: human 0.0, composer 2.1; chords 1/2/3/4: human 0/0/0/0, composer 4/3/0/0
- jacks: human 0%, composer 0%; long notes: human 0%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 10 of 10; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 0.00, composer 1.00
- rhythm (gap to previous chord): human ; composer 1 43%, 1/2 29%, 2 29%
- copies of earlier passages: human ; composer 
- lane motifs only the composer uses here: jumpstream [1 10 4 3]; jumpstream [10 4 3 8]; jumpstream [2 12 1 3]; jumpstream [3 8 12 1]

```
human: Insane          composer: Composer 3.59*
                       448,192,264589,1,0,0:0:0:0:
                       64,192,264824,1,0,0:0:0:0:
                       320,192,264824,1,0,0:0:0:0:
                       192,192,264941,1,0,0:0:0:0:
                       320,192,265058,1,0,0:0:0:0:
                       448,192,265058,1,0,0:0:0:0:
                       64,192,265527,1,0,0:0:0:0:
                       64,192,265761,1,0,0:0:0:0:
                       192,192,265761,1,0,0:0:0:0:
                       448,192,266230,1,0,0:0:0:0:

                    264589 |...#|
                    264824 |#.#.|
                    264941 |.#..|
                    265058 |..##|
                    265527 |#...|
                    265761 |##..|
                    266230 |...#|
```

### 2362621: 88.2-89.7s (#9 intensity 0.00), human MrMoai's Lava Chicken 5.67* vs composer 5.64*, distance 8.85

- notes/s: human 6.1, composer 0.0; chords 1/2/3/4: human 5/2/0/0, composer 0/0/0/0
- jacks: human 44%, composer 0%; long notes: human 0%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 0 of 0; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 1.00, composer 0.00
- rhythm (gap to previous chord): human 1/6 86%, 1/4 14%; composer 
- copies of earlier passages: human ; composer 
- lane motifs only the human uses here: jumpstream [12 1 2 8]; stream [1 2 8 4]; stream [2 8 4 2]; jumpstream [1 2 4 9]

```
human: MrMoai's Lava Chick   composer: Composer 5.67*
64,192,88199,1,2,0:0:0:0:    
192,192,88199,1,4,0:0:0:0:   
448,192,88246,1,0,0:0:0:0:   
320,192,88294,1,0,0:0:0:0:   
64,192,88342,1,0,0:0:0:0:    
192,192,88389,1,0,0:0:0:0:   
320,192,88437,1,0,0:0:0:0:   
64,192,88485,1,0,0:0:0:0:    
448,192,88485,1,0,0:0:0:0:   

  88199 |##..|     
  88246 |...#|     
  88294 |..#.|     
  88342 |#...|     
  88389 |.#..|     
  88437 |..#.|     
  88485 |#..#|     
```

### 1755294: 264.6-269.3s (#30 intensity 0.00), human Normal 2.27* vs composer 2.30*, distance 8.85

- notes/s: human 0.0, composer 1.5; chords 1/2/3/4: human 0/0/0/0, composer 3/2/0/0
- jacks: human 0%, composer 0%; long notes: human 0%, composer 14%
- notes where all human maps rest (break / silence / after the end): composer 7 of 7; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 0.00, composer 1.00
- rhythm (gap to previous chord): human ; composer 2 40%, 1 40%, 3 20%
- copies of earlier passages: human ; composer 
- lane motifs only the composer uses here: jumpstream [2 5 8 6]; jumpstream [10 1 6 8]

```
human: Normal          composer: Composer 2.27*
                       192,192,264824,1,0,0:0:0:0:
                       64,192,265058,1,0,0:0:0:0:
                       320,192,265058,1,0,0:0:0:0:
                       448,192,265527,1,0,0:0:0:0:
                       192,192,265761,1,0,0:0:0:0:
                       320,192,265761,128,0,266230:0:0:0:0:
                       64,192,266230,1,0,0:0:0:0:

                    264824 |.#..|
                    265058 |#.#.|
                    265527 |...#|
                    265761 |.#O.|
                    266230 |#...|
```

### 1579707: 0.0-6.2s (#0 intensity 0.00), human Q U A T T U O R 5.80* vs composer 5.81*, distance 8.44

- notes/s: human 2.7, composer 0.0; chords 1/2/3/4: human 15/1/0/0, composer 0/0/0/0
- jacks: human 0%, composer 0%; long notes: human 100%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 0 of 0; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 1.00, composer 0.00
- rhythm (gap to previous chord): human 1 44%, 1/2 44%, 3/2 6%, rest 6%; composer 
- copies of earlier passages: human ; composer 
- lane motifs only the human uses here: stream [1 4 2 8]; stream [4 2 8 1]; stream [2 8 1 4]; stream [1 8 2 1]

```
human: Q U A T T U O R             composer: Composer 5.80*
64,192,1176,128,0,1369:0:0:0:0:    
320,192,1369,128,0,1563:0:0:0:0:   
192,192,1563,128,0,2724:0:0:0:0:   
448,192,1950,128,0,2143:0:0:0:0:   
64,192,2143,128,0,2337:0:0:0:0:    
320,192,2337,128,0,2530:0:0:0:0:   
448,192,2724,128,0,3498:0:0:0:0:   
320,192,2917,128,0,3498:0:0:0:0:   
64,192,3498,128,0,3885:0:0:0:0:    
192,192,3885,128,0,4272:0:0:0:0:   
320,192,4272,128,0,7369:0:0:0:0:   
448,192,4272,128,0,4466:0:0:0:0:   
64,192,4466,128,0,4659:0:0:0:0:    
192,192,4659,128,0,4853:0:0:0:0:   
448,192,5046,128,0,5434:0:0:0:0:   
192,192,5434,128,0,5627:0:0:0:0:   
448,192,5821,128,0,6305:0:0:0:0:   

   1176 |O...|     
   1369 |..O.|     
   1563 |.O..|     
   1950 |...O|     
   2143 |O...|     
   2337 |..O.|     
   2724 |...O|     
   2917 |..O.|     
   3498 |O...|     
   3885 |.O..|     
   4272 |..OO|     
   4466 |O...|     
   4659 |.O..|     
   5046 |...O|     
   5434 |.O..|     
   5821 |...O|     
```

### 1579707: 0.0-6.2s (#0 intensity 0.00), human K A I S E R 5.21* vs composer 5.21*, distance 8.44

- notes/s: human 2.7, composer 0.0; chords 1/2/3/4: human 15/1/0/0, composer 0/0/0/0
- jacks: human 0%, composer 0%; long notes: human 100%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 0 of 0; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 1.00, composer 0.00
- rhythm (gap to previous chord): human 1 44%, 1/2 44%, 3/2 6%, rest 6%; composer 
- copies of earlier passages: human ; composer 
- lane motifs only the human uses here: stream [1 4 2 8]; stream [4 2 8 1]; stream [2 8 1 4]; stream [1 8 2 1]

```
human: K A I S E R                 composer: Composer 5.21*
64,192,1176,128,0,1369:0:0:0:0:    
320,192,1369,128,0,1563:0:0:0:0:   
192,192,1563,128,0,2724:0:0:0:0:   
448,192,1950,128,0,2143:0:0:0:0:   
64,192,2143,128,0,2337:0:0:0:0:    
320,192,2337,128,0,2530:0:0:0:0:   
448,192,2724,128,0,3498:0:0:0:0:   
320,192,2917,128,0,3498:0:0:0:0:   
64,192,3498,128,0,3885:0:0:0:0:    
192,192,3885,128,0,4272:0:0:0:0:   
320,192,4272,128,0,7369:0:0:0:0:   
448,192,4272,128,0,4466:0:0:0:0:   
64,192,4466,128,0,4659:0:0:0:0:    
192,192,4659,128,0,4853:0:0:0:0:   
448,192,5046,128,0,5434:0:0:0:0:   
192,192,5434,128,0,5627:0:0:0:0:   
448,192,5821,128,0,6305:0:0:0:0:   

   1176 |O...|     
   1369 |..O.|     
   1563 |.O..|     
   1950 |...O|     
   2143 |O...|     
   2337 |..O.|     
   2724 |...O|     
   2917 |..O.|     
   3498 |O...|     
   3885 |.O..|     
   4272 |..OO|     
   4466 |O...|     
   4659 |.O..|     
   5046 |...O|     
   5434 |.O..|     
   5821 |...O|     
```

### 1964340: 0.0-7.4s (#0 intensity 0.00), human GET DOWN [0.85] 6.06* vs composer 6.05*, distance 8.25

- notes/s: human 7.5, composer 0.0; chords 1/2/3/4: human 19/18/0/0, composer 0/0/0/0
- jacks: human 0%, composer 0%; long notes: human 0%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 0 of 0; human break periods: 0, composer: 0
- placement: 0% of composer chords on a human chord time, composer hits 0% of human chords
- distinct 4-chord shapes per chord: human 1.00, composer 0.00
- rhythm (gap to previous chord): human 1/2 76%, 1/4 22%, rest 3%; composer 
- copies of earlier passages: human rhythm 16%; composer 
- lane motifs only the human uses here: jumpstream [4 2 1 12]; jumpstream [3 4 9 2]; jumpstream [2 9 4 3]; jumpstream [9 2 12 2]

```
human: GET DOWN [0.85]      composer: Composer 6.06*
320,192,164,5,0,0:0:0:0:    
64,192,164,1,0,0:0:0:0:     
192,192,389,1,0,0:0:0:0:    
320,192,614,1,0,0:0:0:0:    
448,192,838,1,0,0:0:0:0:    
192,192,1063,1,0,0:0:0:0:   
64,192,1063,1,0,0:0:0:0:    
320,192,1288,1,0,0:0:0:0:   
192,192,1400,1,0,0:0:0:0:   
64,192,1512,1,0,0:0:0:0:    
448,192,1737,1,0,0:0:0:0:   
320,192,1962,1,0,0:0:0:0:   
64,192,1962,1,0,0:0:0:0:    
192,192,2187,1,0,0:0:0:0:   
320,192,2411,1,0,0:0:0:0:   
192,192,2636,1,0,0:0:0:0:   
448,192,2861,1,0,0:0:0:0:   
64,192,2861,1,0,0:0:0:0:    
192,192,3087,1,0,0:0:0:0:   
320,192,3198,1,0,0:0:0:0:   
448,192,3311,1,0,0:0:0:0:   
64,192,3536,1,0,0:0:0:0:    
192,192,3536,1,0,0:0:0:0:   
320,192,3761,1,0,0:0:0:0:   

    164 |#.#.|     
    389 |.#..|     
    614 |..#.|     
    838 |...#|     
   1063 |##..|     
   1288 |..#.|     
   1400 |.#..|     
   1512 |#...|     
   1737 |...#|     
   1962 |#.#.|     
   2187 |.#..|     
   2411 |..#.|     
   2636 |.#..|     
   2861 |#..#|     
   3087 |.#..|     
   3198 |..#.|     
   3311 |...#|     
   3536 |##..|     
   3761 |..#.|     
```


## Best zones

### 872940: 41.1-60.8s (#3 intensity 0.94), human Hard 3.04* vs composer 3.03*, distance 2.12

- notes/s: human 11.4, composer 8.8; chords 1/2/3/4: human 126/50/0/0, composer 116/26/2/0
- jacks: human 0%, composer 0%; long notes: human 4%, composer 4%
- notes where all human maps rest (break / silence / after the end): composer 0 of 174; human break periods: 0, composer: 0
- placement: 88% of composer chords on a human chord time, composer hits 72% of human chords
- distinct 4-chord shapes per chord: human 0.73, composer 0.68
- rhythm (gap to previous chord): human 1/4 74%, 1/2 26%, 1 1%; composer 1/2 56%, 1/4 40%, off 3%, 1/3 1%
- copies of earlier passages: human exact 4%, variation 12%, rhythm 46%; composer exact 5%, rhythm 33%
- lane motifs only the human uses here: jumpstream [12 2 4 9]; jumpstream [3 4 2 9 4 2]; jumpstream [4 2 12 2]; jumpstream [2 12 2 4]
- lane motifs only the composer uses here: stream [1 2 4 8 2 4]; jumpstream [9 4 3 8]; stream [4 2 4 1]; jumpstream [3 8 4 3]

```
human: Hard                          composer: Composer 3.04*
64,192,41058,1,0,0:0:0:0:            64,192,41058,1,0,0:0:0:0:
192,192,41058,1,0,0:0:0:0:           448,192,41058,128,0,41235:0:0:0:0:
320,192,41234,1,0,0:0:0:0:           192,192,41235,1,0,0:0:0:0:
192,192,41323,1,0,0:0:0:0:           320,192,41411,1,0,0:0:0:0:
320,192,41411,1,0,0:0:0:0:           448,192,41411,1,0,0:0:0:0:
448,192,41411,1,0,0:0:0:0:           64,192,41588,1,0,0:0:0:0:
192,192,41587,1,0,0:0:0:0:           192,192,41764,128,0,42029:0:0:0:0:
320,192,41676,1,0,0:0:0:0:           320,192,41941,1,0,0:0:0:0:
448,192,41764,1,0,0:0:0:0:           64,192,42117,1,0,0:0:0:0:
64,192,41764,1,0,0:0:0:0:            448,192,42294,1,0,0:0:0:0:
192,192,41940,1,0,0:0:0:0:           192,192,42470,1,0,0:0:0:0:
320,192,42028,1,0,0:0:0:0:           320,192,42647,1,0,0:0:0:0:
192,192,42117,1,0,0:0:0:0:           64,192,42823,1,0,0:0:0:0:
64,192,42117,1,0,0:0:0:0:            192,192,43000,1,0,0:0:0:0:
448,192,42293,128,0,42470:0:0:0:0:   320,192,43176,1,0,0:0:0:0:
192,192,42470,1,0,0:0:0:0:           448,192,43353,1,0,0:0:0:0:
64,192,42470,1,0,0:0:0:0:            192,192,43529,1,0,0:0:0:0:
320,192,42646,1,0,0:0:0:0:           320,192,43705,1,0,0:0:0:0:
192,192,42734,1,0,0:0:0:0:           448,192,43794,1,0,0:0:0:0:
64,192,42823,1,0,0:0:0:0:            64,192,43882,1,0,0:0:0:0:
448,192,42823,1,0,0:0:0:0:           192,192,43882,1,0,0:0:0:0:
320,192,42999,1,0,0:0:0:0:           320,192,44058,1,0,0:0:0:0:
192,192,43087,1,0,0:0:0:0:           448,192,44147,1,0,0:0:0:0:
320,192,43176,1,0,0:0:0:0:           192,192,44235,1,0,0:0:0:0:

  41058 |##..|       41058 |#..O|
  41234 |..#.|       41235 |.#..|
  41323 |.#..|       41411 |..##|
  41411 |..##|       41588 |#...|
  41587 |.#..|       41764 |.O..|
  41676 |..#.|       41941 |..#.|
  41764 |#..#|       42117 |#...|
  41940 |.#..|       42294 |...#|
  42028 |..#.|       42470 |.#..|
  42117 |##..|       42647 |..#.|
  42293 |...O|       42823 |#...|
  42470 |##..|       43000 |.#..|
  42646 |..#.|       43176 |..#.|
  42734 |.#..|       43353 |...#|
  42823 |#..#|       43529 |.#..|
  42999 |..#.|       43705 |..#.|
  43087 |.#..|       43794 |...#|
  43176 |..#.|       43882 |##..|
                     44058 |..#.|
                     44147 |...#|
                     44235 |.#..|
```

### 872940: 41.1-60.8s (#3 intensity 0.94), human Ash's Easy 1.51* vs composer 1.51*, distance 2.15

- notes/s: human 3.6, composer 2.9; chords 1/2/3/4: human 55/8/0/0, composer 46/6/0/0
- jacks: human 0%, composer 0%; long notes: human 13%, composer 17%
- notes where all human maps rest (break / silence / after the end): composer 0 of 58; human break periods: 1, composer: 0
- placement: 88% of composer chords on a human chord time, composer hits 73% of human chords
- distinct 4-chord shapes per chord: human 0.78, composer 0.86
- rhythm (gap to previous chord): human 1 71%, 1/2 24%, 3/4 3%, 3/2 2%; composer 1 65%, 1/2 19%, 2 13%, 4 2%
- copies of earlier passages: human rhythm 59%; composer 
- lane motifs only the human uses here: stream [8 1 4 2]; jumpstream [3 4 1 2]; jumpstream [2 12 1 8]; stream [1 4 2 1]
- lane motifs only the composer uses here: stream [2 4 1 2]; stream [4 2 8 4 2 8]; stream [4 8 2 4]; jumpstream [2 9 4 2]

```
human: Ash's Easy                    composer: Composer 1.51*
192,192,41058,1,0,0:0:0:0:           320,192,41058,1,0,0:0:0:0:
64,192,41058,1,0,0:0:0:0:            192,192,41411,128,0,42117:0:0:0:0:
320,192,41411,1,0,0:0:0:0:           448,192,42117,128,0,43529:0:0:0:0:
64,192,41764,1,0,0:0:0:0:            64,192,43529,1,0,0:0:0:0:
192,192,42117,1,0,0:0:0:0:           320,192,43882,128,0,44323:0:0:0:0:
320,192,42470,1,0,0:0:0:0:           448,192,44235,1,0,0:0:0:0:
448,192,42470,1,0,0:0:0:0:           192,192,44411,1,0,0:0:0:0:
64,192,42823,1,0,0:0:0:0:            64,192,44588,1,0,0:0:0:0:
448,192,43176,1,0,0:0:0:0:           320,192,44941,1,0,0:0:0:0:
320,192,43528,1,0,0:0:0:0:           448,192,44941,1,0,0:0:0:0:
64,192,43881,1,0,0:0:0:0:            192,192,45294,1,0,0:0:0:0:
192,192,43881,128,0,44234:0:0:0:0:   64,192,45647,1,0,0:0:0:0:
320,192,44234,128,0,44587:0:0:0:0:   320,192,46000,1,0,0:0:0:0:
192,192,44587,1,0,0:0:0:0:           192,192,46353,1,0,0:0:0:0:
448,192,44587,128,0,44940:0:0:0:0:   448,192,46705,1,0,0:0:0:0:
64,192,44940,128,0,45293:0:0:0:0:    320,192,46882,1,0,0:0:0:0:
448,192,45293,1,0,0:0:0:0:           192,192,47058,128,0,47411:0:0:0:0:
192,192,45558,1,0,0:0:0:0:           448,192,47411,1,0,0:0:0:0:
320,192,45823,1,0,0:0:0:0:           320,192,47764,1,0,0:0:0:0:
64,192,45999,1,0,0:0:0:0:            64,192,48117,1,0,0:0:0:0:
448,192,46528,1,0,0:0:0:0:           192,192,48117,1,0,0:0:0:0:
192,192,46705,1,0,0:0:0:0:           448,192,48470,1,0,0:0:0:0:
64,192,46705,1,0,0:0:0:0:            320,192,48823,1,0,0:0:0:0:
320,192,47058,1,0,0:0:0:0:           192,192,49176,1,0,0:0:0:0:

  41058 |##..|       41058 |..#.|
  41411 |..#.|       41411 |.O..|
  41764 |#...|       42117 |...O|
  42117 |.#..|       43529 |#...|
  42470 |..##|       43882 |..O.|
  42823 |#...|       44235 |...#|
  43176 |...#|       44411 |.#..|
  43528 |..#.|       44588 |#...|
  43881 |#O..|       44941 |..##|
  44234 |..O.|       45294 |.#..|
  44587 |.#.O|       45647 |#...|
  44940 |O...|       46000 |..#.|
  45293 |...#|       46353 |.#..|
  45558 |.#..|       46705 |...#|
  45823 |..#.|       46882 |..#.|
  45999 |#...|       47058 |.O..|
  46528 |...#|       47411 |...#|
  46705 |##..|       47764 |..#.|
  47058 |..#.|       48117 |##..|
                     48470 |...#|
                     48823 |..#.|
                     49176 |.#..|
```

### 360565: 74.2-89.2s (#7 intensity 0.85), human MX 3.55* vs composer 3.55*, distance 2.20

- notes/s: human 10.2, composer 9.8; chords 1/2/3/4: human 83/35/0/0, composer 59/38/4/0
- jacks: human 3%, composer 4%; long notes: human 6%, composer 1%
- notes where all human maps rest (break / silence / after the end): composer 0 of 147; human break periods: 0, composer: 0
- placement: 90% of composer chords on a human chord time, composer hits 77% of human chords
- distinct 4-chord shapes per chord: human 0.88, composer 0.73
- rhythm (gap to previous chord): human 1/2 87%, 1/4 7%, 1 6%; composer 1/2 73%, 1/4 12%, 1 9%, 2 5%
- copies of earlier passages: human rhythm 70%; composer exact 7%, rhythm 2%
- lane motifs only the human uses here: jumpstream [8 4 3 4]; jumpstream [2 9 2 4]; jumpstream [2 1 8 3]; jumpstream [1 8 3 4]
- lane motifs only the composer uses here: jumpstream [8 6 1 12]; stream [4 2 1 8]; stream [2 1 8 4]; jumpstream [12 1 6 8]

```
human: MX                            composer: Composer 3.55*
64,192,74201,1,2,0:0:0:65:           448,192,74451,1,0,0:0:0:0:
64,192,74326,1,2,0:0:0:65:           320,192,74701,1,0,0:0:0:0:
448,192,74451,1,8,0:0:0:65:          192,192,74826,1,0,0:0:0:0:
320,192,74513,1,8,0:0:0:65:          64,192,75201,1,0,0:0:0:0:
192,192,74576,1,8,0:0:0:65:          448,192,75326,1,0,0:0:0:0:
64,192,74638,1,8,0:0:0:65:           320,192,75451,128,0,75951:0:0:0:0:
320,192,74701,1,2,0:0:0:65:          192,192,75951,1,0,0:0:0:0:
448,192,74701,1,4,0:0:0:0:           64,192,76201,1,0,0:0:0:0:
192,192,74826,1,0,0:0:0:0:           448,192,76326,1,0,0:0:0:0:
320,192,74951,1,8,0:0:0:65:          192,192,76451,1,0,0:0:0:0:
448,192,74951,1,0,0:0:0:0:           320,192,76451,1,0,0:0:0:0:
192,192,75076,1,0,0:0:0:0:           64,192,76576,1,0,0:0:0:0:
64,192,75201,1,2,0:0:0:65:           320,192,76701,1,0,0:0:0:0:
320,192,75326,1,2,0:0:0:65:          448,192,76701,1,0,0:0:0:0:
64,192,75451,1,8,0:0:0:65:           64,192,76951,1,0,0:0:0:0:
448,192,75451,128,0,75576:0:0:0:0:   192,192,76951,1,0,0:0:0:0:
320,192,75701,1,2,0:0:0:65:          448,192,77076,1,0,0:0:0:0:
192,192,75826,1,2,0:0:0:65:          192,192,77201,1,0,0:0:0:0:
64,192,75951,1,8,0:0:0:65:           320,192,77201,1,0,0:0:0:0:
448,192,75951,1,0,0:0:0:0:           64,192,77326,1,0,0:0:0:0:
192,192,76076,1,2,0:0:0:0:           320,192,77451,1,0,0:0:0:0:
320,192,76201,1,0,0:0:0:0:           448,192,77451,1,0,0:0:0:0:
448,192,76326,1,2,0:0:0:0:           192,192,77576,1,0,0:0:0:0:
64,192,76451,1,8,0:0:0:65:           64,192,77701,1,0,0:0:0:0:

  74201 |#...|       74451 |...#|
  74326 |#...|       74701 |..#.|
  74451 |...#|       74826 |.#..|
  74513 |..#.|       75201 |#...|
  74576 |.#..|       75326 |...#|
  74638 |#...|       75451 |..O.|
  74701 |..##|       75951 |.#..|
  74826 |.#..|       76201 |#...|
  74951 |..##|       76326 |...#|
  75076 |.#..|       76451 |.##.|
  75201 |#...|       76576 |#...|
  75326 |..#.|       76701 |..##|
  75451 |#..O|       76951 |##..|
  75701 |..#.|       77076 |...#|
  75826 |.#..|       77201 |.##.|
  75951 |#..#|       77326 |#...|
  76076 |.#..|       77451 |..##|
  76201 |..#.|       77576 |.#..|
  76326 |...#|       77701 |#...|
  76451 |#...|     
```

### 2362621: 28.8-39.1s (#3 intensity 1.00), human Enma's Hard 3.02* vs composer 3.01*, distance 2.24

- notes/s: human 10.7, composer 10.4; chords 1/2/3/4: human 39/31/3/0, composer 38/30/3/0
- jacks: human 9%, composer 5%; long notes: human 23%, composer 5%
- notes where all human maps rest (break / silence / after the end): composer 0 of 107; human break periods: 0, composer: 0
- placement: 85% of composer chords on a human chord time, composer hits 82% of human chords
- distinct 4-chord shapes per chord: human 0.83, composer 0.91
- rhythm (gap to previous chord): human 1/2 81%, 1/4 14%, 3/4 3%, 1 3%; composer 1/2 73%, 1/4 17%, 1 10%
- copies of earlier passages: human variation 9%, rhythm 36%; composer variation 12%
- lane motifs only the human uses here: jumpstream [8 4 3 4]; jumpstream [3 4 10 1]; jumpstream [3 4 8 1]; jumpstream [3 4 8 1 2 12]
- lane motifs only the composer uses here: jumpstream [1 12 3 4]; jumpstream [13 2 12 1]; jumpstream [1 10 4 9]; chord stairs [10 4 9 2]

```
human: Enma's Hard                             composer: Composer 3.02*
448,192,28913,1,0,0:0:0:70:HiHat.wav           320,192,28771,1,0,0:0:0:0:
64,192,29056,1,2,0:0:0:0:                      448,192,28771,1,0,0:0:0:0:
192,192,29056,128,0,29199:0:0:0:0:             64,192,28914,1,0,0:0:0:0:
320,192,29199,1,0,0:0:0:70:HiHat.wav           192,192,29056,1,0,0:0:0:0:
64,192,29342,1,8,0:0:0:0:                      448,192,29056,1,0,0:0:0:0:
448,192,29342,128,2,29485:0:0:0:0:             320,192,29199,1,0,0:0:0:0:
192,192,29485,1,0,0:0:0:70:HiHat.wav           64,192,29342,1,0,0:0:0:0:
320,192,29627,1,2,0:0:0:0:                     192,192,29342,1,0,0:0:0:0:
448,192,29627,1,0,0:0:0:0:                     448,192,29485,1,0,0:0:0:0:
64,192,29770,1,0,0:0:0:0:                      64,192,29628,1,0,0:0:0:0:
192,192,29913,1,2,0:0:0:0:                     320,192,29628,1,0,0:0:0:0:
448,192,29913,1,0,0:0:0:70:beep.wav            192,192,29699,1,0,0:0:0:0:
64,192,30056,1,2,0:0:0:0:                      320,192,29771,1,0,0:0:0:0:
320,192,30056,1,0,0:0:0:0:                     448,192,29842,1,0,0:0:0:0:
64,192,30199,1,2,0:0:0:0:                      64,192,29914,1,0,0:0:0:0:
192,192,30199,1,4,0:0:0:0:                     192,192,29914,1,0,0:0:0:0:
320,192,30342,1,2,0:0:0:0:                     320,192,30056,1,0,0:0:0:0:
448,192,30342,1,4,0:0:0:0:                     64,192,30199,1,0,0:0:0:0:
64,192,30485,1,8,0:0:0:0:                      448,192,30199,1,0,0:0:0:0:
192,192,30485,1,2,0:0:0:0:                     192,192,30485,1,0,0:0:0:0:
448,192,30485,1,0,0:0:0:0:                     320,192,30485,128,0,30843:0:0:0:0:
320,192,30627,128,0,30770:0:0:0:70:HiHat.wav   448,192,30771,1,0,0:0:0:0:
64,192,30770,1,2,0:0:0:0:                      64,192,30914,1,0,0:0:0:0:
448,192,30770,1,0,0:0:0:0:                     192,192,31056,1,0,0:0:0:0:

  28913 |...#|       28771 |..##|
  29056 |#O..|       28914 |#...|
  29199 |..#.|       29056 |.#.#|
  29342 |#..O|       29199 |..#.|
  29485 |.#..|       29342 |##..|
  29627 |..##|       29485 |...#|
  29770 |#...|       29628 |#.#.|
  29913 |.#.#|       29699 |.#..|
  30056 |#.#.|       29771 |..#.|
  30199 |##..|       29842 |...#|
  30342 |..##|       29914 |##..|
  30485 |##.#|       30056 |..#.|
  30627 |..O.|       30199 |#..#|
  30770 |#..#|       30485 |.#O.|
                     30771 |...#|
                     30914 |#...|
                     31056 |.#..|
```

### 360565: 211.2-227.2s (#22 intensity 0.98), human NM 2.00* vs composer 2.02*, distance 2.27

- notes/s: human 4.9, composer 5.1; chords 1/2/3/4: human 49/15/0/0, composer 33/23/1/0
- jacks: human 0%, composer 0%; long notes: human 3%, composer 2%
- notes where all human maps rest (break / silence / after the end): composer 0 of 82; human break periods: 0, composer: 0
- placement: 82% of composer chords on a human chord time, composer hits 73% of human chords
- distinct 4-chord shapes per chord: human 0.84, composer 0.91
- rhythm (gap to previous chord): human 1 72%, 1/2 17%, 3/2 6%, 2 5%; composer 1 82%, 3/2 5%, 2 5%, 1/2 4%
- copies of earlier passages: human exact 13%, rhythm 67%; composer rhythm 11%
- lane motifs only the human uses here: jumpstream [12 1 6 8]; roll [8 4 2 4]; stream [4 2 4 1]; stream [8 4 1 4]
- lane motifs only the composer uses here: jumpstream [12 1 10 4]; jumpstream [4 3 8 5]; jumpstream [2 9 4 2]; jumpstream [9 4 2 9]

```
human: NM                               composer: Composer 2.00*
64,192,211451,1,8,0:0:0:65:             448,192,211201,1,0,0:0:0:0:
192,192,211576,1,0,0:0:0:0:             64,192,211451,1,0,0:0:0:0:
320,192,211701,1,2,0:0:0:65:            192,192,211451,1,0,0:0:0:0:
192,192,211951,1,8,0:0:0:65:            320,192,211701,1,0,0:0:0:0:
448,192,212201,1,0,0:0:0:0:             64,192,211951,1,0,0:0:0:0:
64,192,212451,1,8,0:0:0:65:             448,192,211951,1,0,0:0:0:0:
192,192,212576,1,0,0:0:0:0:             64,192,212201,1,0,0:0:0:0:
448,192,212701,1,2,0:0:0:65:            192,192,212201,1,0,0:0:0:0:
192,192,212951,1,8,0:0:0:65:            448,192,212201,1,0,0:0:0:0:
448,192,213076,1,0,0:0:0:0:             320,192,212326,1,0,0:0:0:0:
320,192,213326,1,2,0:0:0:65:            64,192,212451,1,0,0:0:0:0:
64,192,213451,1,8,0:0:0:65:             192,192,212451,1,0,0:0:0:0:
192,192,213701,1,2,0:0:0:65:            320,192,212701,1,0,0:0:0:0:
320,192,213951,1,8,0:0:0:65:            448,192,212701,1,0,0:0:0:0:
448,192,214201,1,0,0:0:0:0:             64,192,212951,1,0,0:0:0:0:
64,192,214451,1,8,0:0:0:65:             192,192,213201,1,0,0:0:0:0:
64,192,214701,1,2,0:0:0:0:              448,192,213201,1,0,0:0:0:0:
64,192,215201,1,2,0:0:0:65:             320,192,213451,1,0,0:0:0:0:
320,192,215451,1,8,0:0:0:65:            64,192,213701,1,0,0:0:0:0:
448,192,215701,128,2,216076:0:0:0:65:   192,192,213701,1,0,0:0:0:0:
192,192,216076,128,2,216451:0:0:0:65:   448,192,213951,1,0,0:0:0:0:
320,192,216451,1,12,0:0:0:65:           64,192,214201,1,0,0:0:0:0:
64,192,216701,1,2,0:0:0:65:             320,192,214201,1,0,0:0:0:0:
448,192,216951,1,8,0:0:0:65:            192,192,214451,1,0,0:0:0:0:

 211451 |#...|      211201 |...#|
 211576 |.#..|      211451 |##..|
 211701 |..#.|      211701 |..#.|
 211951 |.#..|      211951 |#..#|
 212201 |...#|      212201 |##.#|
 212451 |#...|      212326 |..#.|
 212576 |.#..|      212451 |##..|
 212701 |...#|      212701 |..##|
 212951 |.#..|      212951 |#...|
 213076 |...#|      213201 |.#.#|
 213326 |..#.|      213451 |..#.|
 213451 |#...|      213701 |##..|
 213701 |.#..|      213951 |...#|
 213951 |..#.|      214201 |#.#.|
 214201 |...#|      214451 |.#..|
 214451 |#...|     
 214701 |#...|     
 215201 |#...|     
 215451 |..#.|     
 215701 |...O|     
 216076 |.O..|     
 216451 |..#.|     
 216701 |#...|     
 216951 |...#|     
```

### 2400881: 15.3-26.9s (#2 intensity 0.98), human Hard 3.05* vs composer 3.06*, distance 2.29

- notes/s: human 7.9, composer 8.2; chords 1/2/3/4: human 54/19/0/0, composer 36/25/3/0
- jacks: human 0%, composer 1%; long notes: human 9%, composer 2%
- notes where all human maps rest (break / silence / after the end): composer 0 of 95; human break periods: 0, composer: 0
- placement: 89% of composer chords on a human chord time, composer hits 78% of human chords
- distinct 4-chord shapes per chord: human 0.81, composer 0.89
- rhythm (gap to previous chord): human 1/2 53%, 1/4 47%; composer 1/2 67%, 1/4 31%, 1 2%
- copies of earlier passages: human exact 10%, rhythm 21%; composer 
- lane motifs only the human uses here: jumpstream [1 12 2 1]; jumpstream [12 2 1 2]; roll [2 1 2 4]; stream [1 2 8 4]
- lane motifs only the composer uses here: jumpstream [8 3 4 9]; jumpstream [9 6 8 3]; jumpstream [12 2 9 6]; chord stairs [4 9 6 1]

```
human: Hard                  composer: Composer 3.05*
64,192,15277,1,0,0:0:0:0:    448,192,15277,1,0,0:0:0:0:
320,192,15484,1,0,0:0:0:0:   64,192,15484,1,0,0:0:0:0:
192,192,15484,1,0,0:0:0:0:   320,192,15691,1,0,0:0:0:0:
64,192,15690,1,0,0:0:0:0:    448,192,15691,1,0,0:0:0:0:
320,192,15897,1,0,0:0:0:0:   192,192,15898,1,0,0:0:0:0:
448,192,15897,1,0,0:0:0:0:   64,192,16105,1,0,0:0:0:0:
192,192,16001,1,0,0:0:0:0:   448,192,16105,1,0,0:0:0:0:
64,192,16104,1,0,0:0:0:0:    320,192,16312,1,0,0:0:0:0:
192,192,16208,1,0,0:0:0:0:   64,192,16519,1,0,0:0:0:0:
320,192,16311,1,0,0:0:0:0:   192,192,16519,1,0,0:0:0:0:
192,192,16518,1,0,0:0:0:0:   448,192,16622,1,0,0:0:0:0:
448,192,16725,1,0,0:0:0:0:   320,192,16725,1,0,0:0:0:0:
320,192,16725,1,0,0:0:0:0:   64,192,16932,1,0,0:0:0:0:
192,192,16932,1,0,0:0:0:0:   192,192,16932,1,0,0:0:0:0:
448,192,17139,1,0,0:0:0:0:   448,192,17139,1,0,0:0:0:0:
320,192,17139,1,0,0:0:0:0:   64,192,17346,1,0,0:0:0:0:
192,192,17346,1,0,0:0:0:0:   320,192,17346,1,0,0:0:0:0:
448,192,17553,1,0,0:0:0:0:   192,192,17450,1,0,0:0:0:0:
64,192,17553,1,0,0:0:0:0:    448,192,17553,1,0,0:0:0:0:
320,192,17656,1,0,0:0:0:0:   64,192,17760,1,0,0:0:0:0:
192,192,17759,1,0,0:0:0:0:   320,192,17760,1,0,0:0:0:0:
320,192,17863,1,0,0:0:0:0:   64,192,17967,1,0,0:0:0:0:
448,192,17966,1,0,0:0:0:0:   320,192,17967,1,0,0:0:0:0:
64,192,18173,1,0,0:0:0:0:    448,192,17967,1,0,0:0:0:0:

  15277 |#...|       15277 |...#|
  15484 |.##.|       15484 |#...|
  15690 |#...|       15691 |..##|
  15897 |..##|       15898 |.#..|
  16001 |.#..|       16105 |#..#|
  16104 |#...|       16312 |..#.|
  16208 |.#..|       16519 |##..|
  16311 |..#.|       16622 |...#|
  16518 |.#..|       16725 |..#.|
  16725 |..##|       16932 |##..|
  16932 |.#..|       17139 |...#|
  17139 |..##|       17346 |#.#.|
  17346 |.#..|       17450 |.#..|
  17553 |#..#|       17553 |...#|
  17656 |..#.|       17760 |#.#.|
  17759 |.#..|       17967 |#.##|
  17863 |..#.|     
  17966 |...#|     
  18173 |#...|     
```

### 1579707: 137.8-150.2s (#12 intensity 1.00), human Q U A T T U O R 5.80* vs composer 5.81*, distance 2.31

- notes/s: human 19.1, composer 19.3; chords 1/2/3/4: human 106/50/10/0, composer 112/35/15/3
- jacks: human 10%, composer 21%; long notes: human 2%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 0 of 239; human break periods: 0, composer: 0
- placement: 83% of composer chords on a human chord time, composer hits 83% of human chords
- distinct 4-chord shapes per chord: human 0.67, composer 0.75
- rhythm (gap to previous chord): human 1/4 51%, 1/8 48%, 1/2 1%; composer 1/4 48%, 1/8 44%, off 7%
- copies of earlier passages: human exact 4%, mirror 3%, rhythm 46%; composer variation 4%, rhythm 4%
- lane motifs only the human uses here: stream [1 4 8 2]; stream [4 8 2 1]; stream [1 2 8 4]; jumptrill [3 12 3 12]
- lane motifs only the composer uses here: jumpstream [2 12 1 2]; jumpstream [4 11 4 2]; jumpstream [13 2 4 1]; stream [2 4 1 2]

```
human: Q U A T T U O R        composer: Composer 5.80*
320,192,137821,1,0,0:0:0:0:   192,192,137821,1,0,0:0:0:0:
448,192,137821,1,0,0:0:0:0:   320,192,137821,1,0,0:0:0:0:
64,192,137821,1,0,0:0:0:0:    448,192,137870,1,0,0:0:0:0:
192,192,137917,1,0,0:0:0:0:   64,192,137918,1,0,0:0:0:0:
64,192,138014,1,0,0:0:0:0:    320,192,138015,1,0,0:0:0:0:
448,192,138014,1,0,0:0:0:0:   192,192,138111,1,0,0:0:0:0:
320,192,138111,1,0,0:0:0:0:   64,192,138208,1,0,0:0:0:0:
64,192,138208,1,0,0:0:0:0:    320,192,138208,1,0,0:0:0:0:
192,192,138208,1,0,0:0:0:0:   448,192,138208,1,0,0:0:0:0:
448,192,138208,1,0,0:0:0:0:   192,192,138257,1,0,0:0:0:0:
320,192,138305,1,0,0:0:0:0:   320,192,138317,1,0,0:0:0:0:
448,192,138401,1,0,0:0:0:0:   64,192,138402,1,0,0:0:0:0:
64,192,138401,1,0,0:0:0:0:    192,192,138499,1,0,0:0:0:0:
192,192,138498,1,0,0:0:0:0:   320,192,138499,1,0,0:0:0:0:
64,192,138595,1,0,0:0:0:0:    64,192,138595,1,0,0:0:0:0:
448,192,138595,1,0,0:0:0:0:   448,192,138595,1,0,0:0:0:0:
320,192,138595,1,0,0:0:0:0:   320,192,138644,1,0,0:0:0:0:
192,192,138692,1,0,0:0:0:0:   192,192,138692,1,0,0:0:0:0:
64,192,138788,1,0,0:0:0:0:    64,192,138789,1,0,0:0:0:0:
320,192,138788,1,0,0:0:0:0:   320,192,138886,1,0,0:0:0:0:
448,192,138885,1,0,0:0:0:0:   448,192,138886,1,0,0:0:0:0:
192,192,138982,1,0,0:0:0:0:   64,192,138982,1,0,0:0:0:0:
320,192,138982,1,0,0:0:0:0:   192,192,138982,1,0,0:0:0:0:
64,192,139030,1,0,0:0:0:0:    448,192,138982,1,0,0:0:0:0:

 137821 |#.##|      137821 |.##.|
 137917 |.#..|      137870 |...#|
 138014 |#..#|      137918 |#...|
 138111 |..#.|      138015 |..#.|
 138208 |##.#|      138111 |.#..|
 138305 |..#.|      138208 |#.##|
 138401 |#..#|      138257 |.#..|
 138498 |.#..|      138317 |..#.|
 138595 |#.##|      138402 |#...|
 138692 |.#..|      138499 |.##.|
 138788 |#.#.|      138595 |#..#|
 138885 |...#|      138644 |..#.|
 138982 |.##.|      138692 |.#..|
 139030 |#...|      138789 |#...|
                    138886 |..##|
                    138982 |##.#|
```

### 872940: 12.8-35.4s (#1 intensity 0.94), human Hard 3.04* vs composer 3.03*, distance 2.37

- notes/s: human 11.0, composer 9.1; chords 1/2/3/4: human 107/71/0/0, composer 99/46/5/0
- jacks: human 0%, composer 0%; long notes: human 19%, composer 5%
- notes where all human maps rest (break / silence / after the end): composer 0 of 206; human break periods: 0, composer: 0
- placement: 91% of composer chords on a human chord time, composer hits 77% of human chords
- distinct 4-chord shapes per chord: human 0.69, composer 0.67
- rhythm (gap to previous chord): human 1/4 57%, 1/2 43%; composer 1/2 71%, 1/4 29%
- copies of earlier passages: human variation 16%, rhythm 39%; composer variation 9%, rhythm 23%
- lane motifs only the human uses here: jumpstream [3 8 4 3]; jumpstream [12 2 4 9]; jumpstream [2 12 2 4]; jumpstream [3 4 8 3]
- lane motifs only the composer uses here: jumpstream [2 9 4 2 9 4]; jumpstream [9 2 4 9 2 4]; jumpstream [3 4 9 2]; jumpstream [4 2 9 4 2 9]

```
human: Hard                          composer: Composer 3.04*
64,192,12823,1,0,0:0:0:0:            320,192,12823,1,0,0:0:0:0:
448,192,12823,128,0,12999:0:0:0:0:   192,192,13000,1,0,0:0:0:0:
192,192,12999,1,0,0:0:0:0:           64,192,13176,1,0,0:0:0:0:
64,192,13087,1,0,0:0:0:0:            320,192,13353,1,0,0:0:0:0:
448,192,13176,1,0,0:0:0:0:           192,192,13529,1,0,0:0:0:0:
320,192,13176,128,0,13352:0:0:0:0:   64,192,13705,1,0,0:0:0:0:
64,192,13352,1,0,0:0:0:0:            448,192,13882,1,0,0:0:0:0:
192,192,13440,1,0,0:0:0:0:           320,192,14058,1,0,0:0:0:0:
320,192,13528,1,0,0:0:0:0:           192,192,14235,1,0,0:0:0:0:
448,192,13528,1,0,0:0:0:0:           448,192,14411,1,0,0:0:0:0:
192,192,13705,1,0,0:0:0:0:           64,192,14588,1,0,0:0:0:0:
64,192,13793,1,0,0:0:0:0:            320,192,14588,1,0,0:0:0:0:
320,192,13881,1,0,0:0:0:0:           192,192,14764,1,0,0:0:0:0:
448,192,13881,1,0,0:0:0:0:           64,192,14941,1,0,0:0:0:0:
64,192,14058,128,0,14234:0:0:0:0:    448,192,14941,1,0,0:0:0:0:
320,192,14234,1,0,0:0:0:0:           320,192,15117,1,0,0:0:0:0:
448,192,14234,128,0,14411:0:0:0:0:   64,192,15294,1,0,0:0:0:0:
192,192,14411,1,0,0:0:0:0:           192,192,15294,1,0,0:0:0:0:
320,192,14499,1,0,0:0:0:0:           320,192,15470,1,0,0:0:0:0:
64,192,14587,1,0,0:0:0:0:            64,192,15647,1,0,0:0:0:0:
192,192,14587,128,0,14764:0:0:0:0:   448,192,15647,1,0,0:0:0:0:
448,192,14764,1,0,0:0:0:0:           192,192,15823,1,0,0:0:0:0:
320,192,14852,1,0,0:0:0:0:           320,192,16000,1,0,0:0:0:0:
192,192,14940,1,0,0:0:0:0:           448,192,16000,1,0,0:0:0:0:

  12823 |#..O|       12823 |..#.|
  12999 |.#..|       13000 |.#..|
  13087 |#...|       13176 |#...|
  13176 |..O#|       13353 |..#.|
  13352 |#...|       13529 |.#..|
  13440 |.#..|       13705 |#...|
  13528 |..##|       13882 |...#|
  13705 |.#..|       14058 |..#.|
  13793 |#...|       14235 |.#..|
  13881 |..##|       14411 |...#|
  14058 |O...|       14588 |#.#.|
  14234 |..#O|       14764 |.#..|
  14411 |.#..|       14941 |#..#|
  14499 |..#.|       15117 |..#.|
  14587 |#O..|       15294 |##..|
  14764 |...#|       15470 |..#.|
  14852 |..#.|       15647 |#..#|
  14940 |.#..|       15823 |.#..|
                     16000 |..##|
```

### 360565: 112.2-125.2s (#11 intensity 0.94), human MX 3.55* vs composer 3.55*, distance 2.37

- notes/s: human 12.0, composer 12.0; chords 1/2/3/4: human 82/37/0/0, composer 50/44/6/0
- jacks: human 1%, composer 6%; long notes: human 3%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 0 of 156; human break periods: 0, composer: 0
- placement: 91% of composer chords on a human chord time, composer hits 76% of human chords
- distinct 4-chord shapes per chord: human 0.74, composer 0.76
- rhythm (gap to previous chord): human 1/2 66%, 1/3 18%, 1/4 13%, 2/3 2%; composer 1/2 91%, 1 5%, 1/4 4%
- copies of earlier passages: human exact 7%, rhythm 65%; composer exact 5%, mirror 6%, variation 5%, rhythm 26%
- lane motifs only the human uses here: stream [4 1 2 8]; stream [1 2 8 4]; jumpstream [8 4 3 8]; stream [4 1 2 8 4 1]
- lane motifs only the composer uses here: jumpstream [6 1 12 2]; jumpstream [8 6 1 12]; jumpstream [12 3 12 1]; jumpstream [8 3 4 9]

```
human: MX                              composer: Composer 3.55*
64,192,112201,1,0,0:0:0:0:             64,192,112201,1,0,0:0:0:0:
192,192,112326,1,2,0:0:0:65:           448,192,112201,1,0,0:0:0:0:
320,192,112451,1,8,0:0:0:65:           192,192,112326,1,0,0:0:0:0:
448,192,112451,1,4,0:0:0:0:            320,192,112326,1,0,0:0:0:0:
64,192,112576,1,0,0:0:0:0:             448,192,112326,1,0,0:0:0:0:
192,192,112701,1,2,0:0:0:65:           192,192,112451,1,0,0:0:0:0:
448,192,112701,128,0,113201:0:0:0:0:   320,192,112451,1,0,0:0:0:0:
320,192,112826,1,0,0:0:0:0:            64,192,112576,1,0,0:0:0:0:
64,192,112951,1,8,0:0:0:65:            320,192,112701,1,0,0:0:0:0:
192,192,112951,1,0,0:0:0:0:            448,192,112701,1,0,0:0:0:0:
320,192,113076,1,0,0:0:0:0:            192,192,112951,1,0,0:0:0:0:
64,192,113201,128,2,113701:0:0:0:0:    64,192,113201,1,0,0:0:0:0:
192,192,113201,1,0,0:0:0:0:            448,192,113326,1,0,0:0:0:0:
448,192,113326,1,2,0:0:0:65:           192,192,113451,1,0,0:0:0:0:
192,192,113451,1,8,0:0:0:65:           320,192,113451,1,0,0:0:0:0:
320,192,113451,1,0,0:0:0:0:            64,192,113576,1,0,0:0:0:0:
448,192,113576,1,0,0:0:0:0:            320,192,113701,1,0,0:0:0:0:
320,192,113701,1,2,0:0:0:65:           448,192,113701,1,0,0:0:0:0:
192,192,113826,1,2,0:0:0:65:           192,192,113826,1,0,0:0:0:0:
64,192,113951,1,8,0:0:0:65:            64,192,113951,1,0,0:0:0:0:
448,192,113951,1,0,0:0:0:0:            320,192,113951,1,0,0:0:0:0:
320,192,114076,1,2,0:0:0:0:            448,192,113951,1,0,0:0:0:0:
64,192,114201,1,8,0:0:0:65:            64,192,114076,1,0,0:0:0:0:
192,192,114263,1,8,0:0:0:65:           192,192,114076,1,0,0:0:0:0:

 112201 |#...|      112201 |#..#|
 112326 |.#..|      112326 |.###|
 112451 |..##|      112451 |.##.|
 112576 |#...|      112576 |#...|
 112701 |.#.O|      112701 |..##|
 112826 |..#.|      112951 |.#..|
 112951 |##..|      113201 |#...|
 113076 |..#.|      113326 |...#|
 113201 |O#..|      113451 |.##.|
 113326 |...#|      113576 |#...|
 113451 |.##.|      113701 |..##|
 113576 |...#|      113826 |.#..|
 113701 |..#.|      113951 |#.##|
 113826 |.#..|      114076 |##..|
 113951 |#..#|     
 114076 |..#.|     
 114201 |#...|     
 114263 |.#..|     
```

### 2400881: 15.3-26.9s (#2 intensity 0.98), human Insane 3.87* vs composer 3.90*, distance 2.39

- notes/s: human 11.6, composer 11.0; chords 1/2/3/4: human 47/42/1/0, composer 54/29/5/0
- jacks: human 1%, composer 3%; long notes: human 3%, composer 0%
- notes where all human maps rest (break / silence / after the end): composer 0 of 127; human break periods: 0, composer: 0
- placement: 81% of composer chords on a human chord time, composer hits 79% of human chords
- distinct 4-chord shapes per chord: human 0.87, composer 0.89
- rhythm (gap to previous chord): human 1/4 76%, 1/2 24%; composer 1/4 65%, 1/2 31%, 1/8 5%
- copies of earlier passages: human variation 7%, rhythm 22%; composer 
- lane motifs only the human uses here: jumpstream [3 12 1 12]; jumpstream [3 8 3 12]; jumpstream [2 1 2 12]; jumpstream [8 4 3 8]
- lane motifs only the composer uses here: jumpstream [4 8 3 4]; jumpstream [1 12 2 1]; jumpstream [8 3 4 9]; jumpstream [8 6 1 12]

```
human: Insane                composer: Composer 3.87*
192,192,15277,1,0,0:0:0:0:   192,192,15277,1,0,0:0:0:0:
192,192,15484,1,0,0:0:0:0:   448,192,15277,1,0,0:0:0:0:
448,192,15484,1,0,0:0:0:0:   64,192,15381,1,0,0:0:0:0:
64,192,15690,1,0,0:0:0:0:    320,192,15484,1,0,0:0:0:0:
448,192,15897,1,0,0:0:0:0:   448,192,15588,1,0,0:0:0:0:
320,192,15897,1,0,0:0:0:0:   64,192,15691,1,0,0:0:0:0:
192,192,16001,1,0,0:0:0:0:   192,192,15691,1,0,0:0:0:0:
64,192,16001,1,0,0:0:0:0:    320,192,15898,1,0,0:0:0:0:
448,192,16104,1,0,0:0:0:0:   448,192,16001,1,0,0:0:0:0:
192,192,16208,1,0,0:0:0:0:   64,192,16105,1,0,0:0:0:0:
64,192,16208,1,0,0:0:0:0:    192,192,16105,1,0,0:0:0:0:
320,192,16311,1,0,0:0:0:0:   320,192,16208,1,0,0:0:0:0:
448,192,16311,1,0,0:0:0:0:   64,192,16312,1,0,0:0:0:0:
192,192,16518,1,0,0:0:0:0:   448,192,16312,1,0,0:0:0:0:
448,192,16725,1,0,0:0:0:0:   192,192,16519,1,0,0:0:0:0:
192,192,16725,1,0,0:0:0:0:   320,192,16519,1,0,0:0:0:0:
64,192,16932,1,0,0:0:0:0:    448,192,16622,1,0,0:0:0:0:
64,192,17139,1,0,0:0:0:0:    64,192,16725,1,0,0:0:0:0:
320,192,17139,1,0,0:0:0:0:   192,192,16932,1,0,0:0:0:0:
192,192,17346,1,0,0:0:0:0:   320,192,16932,1,0,0:0:0:0:
320,192,17553,1,0,0:0:0:0:   448,192,16932,1,0,0:0:0:0:
448,192,17553,1,0,0:0:0:0:   64,192,17036,1,0,0:0:0:0:
64,192,17656,1,0,0:0:0:0:    192,192,17139,1,0,0:0:0:0:
192,192,17656,1,0,0:0:0:0:   64,192,17346,1,0,0:0:0:0:

  15277 |.#..|       15277 |.#.#|
  15484 |.#.#|       15381 |#...|
  15690 |#...|       15484 |..#.|
  15897 |..##|       15588 |...#|
  16001 |##..|       15691 |##..|
  16104 |...#|       15898 |..#.|
  16208 |##..|       16001 |...#|
  16311 |..##|       16105 |##..|
  16518 |.#..|       16208 |..#.|
  16725 |.#.#|       16312 |#..#|
  16932 |#...|       16519 |.##.|
  17139 |#.#.|       16622 |...#|
  17346 |.#..|       16725 |#...|
  17553 |..##|       16932 |.###|
  17656 |##..|       17036 |#...|
                     17139 |.#..|
                     17346 |#...|
```
