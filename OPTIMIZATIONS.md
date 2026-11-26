# Drone 1 speed optimizations

A log of changes made to speed up the first drone's route (`drone_wall_ride`, clues 5 - 8),
with the test results behind each one. Newest at the bottom.

## How it was measured

Every batch is full competition runs (`competition.launch.py`, both drones) in a fresh,
isolated simulation, with all 8 clue boards placed at random inside the competition's
ranges. Times are **sim seconds on the competition timer** (start to stop message). Phase
times come from drone 1's state changes; true positions come from Gazebo's model pose.
5 runs per batch (baseline: 10), so differences under ~3 s are within run-to-run noise.

## Baseline (10 runs)

| | mean | range |
|---|---|---|
| Timer | **106.6 s** | 101.4 - 120.0 |
| Wall rides (south, east, north to the tunnel) | 54.8 s | 46.9 - 58.9 |
| Climb + cross to the plateau | 22.6 s | 20.0 - 24.7 |
| Clue 8 search | 10.2 s | 4.7 - 19.3 |

Clues right: 75/80 (clue 4: 8/10, clue 6: 7/10, everything else 10/10).

## 1. Climb and cross at the same time

**Before:** climb straight up to 2.3 m in front of the tunnel (~12.7 s), *then* fly to the
plateau (~9.9 s).

**Change:** the cross starts once the drone is 1.0 m up and the climb carries on during it.
1.0 m clears the tunnel (0.62 m tall) and is where the optical flow keeps working while
climbing. Until the drone is within 0.1 m of cruise height it is held east of x = 2.8, so it
can't reach the plateau's slope (starts at x ≈ 2.4) too low.

**Problem found and fixed:** the first test batch read clue 8 in only 1 of 5 runs. Climbing
and crossing at once, dead reckoning came up ~1 m short. The cross ended above the plateau's
sloped edge, where the range reading flickers, and the clue 8 search mistook that for the
plateau's far edge and stopped early. (The baseline was already marginal here: 2 of 10 runs
started the search on the edge and got lucky.) Now the cross ends only once the downward
range sensor has had the plateau under the drone for 1 s, carrying on up to 1 m past the
dead-reckoned point if needed. In testing it always found the plateau 0.36 - 0.48 m past
that point.

**Also fixed (unrelated bug found in testing):** pressed nose-first into a wall, the front
range sensor sits *inside* the wall and reads "nothing ahead". The drone then never
recognised the corner and pushed into the east wall for the rest of the run. After a close
reading (< 0.35 m), "nothing" now counts as touching until a real reading returns.

**Parameters:** `cross_climb_altitude` 1.0, `plateau_edge_x` 2.8, `plateau_confirm_time` 1.0,
`cross_overshoot` 1.0.

**Result (5 runs):** timer **95.6 s** (89.7 - 101.8), **−11 s**. Climb + cross 15.7 s
(−6.9 s). Clue 8 5/5. **Kept.**

## 2. Shorter slowdown after a board leaves view

**Planned:** slow down only for boards close enough to read, not every board in view.

**What the data showed:** sightings of drone 1's boards during the wall rides are already
almost all readable size (≥ 1% of the image, or a side-camera view of the board being
passed; clue 7 reads fine at 0.8%). There was little to gain from a size threshold. The
waste was the 1.5 s the drone stayed at cruise speed *after* the last sighting. Boards are
seen ~5 times a second and only 6 of 988 gaps between sightings were over 0.5 s.

**Change:** `board_slow_time` 1.5 → 0.6 s.

**Result (5 runs):** no measurable change. Wall rides 55.6 s against 54.8 s, within noise and
inflated by one run that missed clue 5 (see below). Expected gain is ~1 - 2 s per run.
Kept because reading wasn't affected (clue 5 still got 2 - 4 views when seen).

## 3. Clue 8 search that doesn't stall

**Planned:** fly faster until clue 8 is first seen, then slow down.

**What the data showed:** after change 1, clue 8 is already in view (1.1% of the image)
when the search starts, so "until first seen" gains nothing. The time was lost to a stall
seen in every run: the board's size stayed at ~1.4% for 4 - 5 s while the drone drifted
~0.2 m *back* east (true position) at the plateau's edge. The search used a plain speed loop
at 0.15 m/s, whose push (~0.15 m/s²) is far below the 1.8 m/s² gusts, so a gust plus a
slightly-off velocity estimate stopped it and nothing built up to recover.

**Change:** the search now holds a point moving west, like the cross does: falling behind
means a growing push. The point moves at 0.3 m/s while the board covers under 2% of the
image, then 0.15 m/s for the last part. It never gets more than 0.4 m ahead of the drone.

**Parameters:** `clue8_far_speed` 0.3, `clue8_near_area` 0.02, `clue8_lead` 0.4
(`clue8_speed` 0.15 unchanged).

**Result (5 runs):** clue 8 search **6.2 s** (5.3 - 8.7) against 7.8 s after change 1 and
10.2 s (up to 19.3) in the baseline. Clue 8 5/5. **Kept.**

## Where it stands (changes 1 - 3)

| | Baseline | Now (5 runs) |
|---|---|---|
| Timer, all runs | 106.6 s | 97.4 s (91.4 - 107.9) |
| Timer, runs where clue 5 was read | 106.6 s | **94.8 s** (91.4 - 103.7) |
| Climb + cross | 22.6 s | 15.8 s |
| Clue 8 search | 10.2 s | 6.2 s |
| Wall rides | 54.8 s | 55.8 s |

About **10 - 12 s faster**. All 15 runs after the fixes read clue 8.

## Open issues found while testing (not fixed)

- **Clue 5 is missed when its board is placed close to the wall.** If the board ends up less
  than ~0.62 m from the drone's wall line (board y below about −1.75, depending on the
  drone's wobble), it doesn't fit in the side camera and is never seen. That happened in 5 of
  20 test runs; it never happened in the baseline batch only because no board landed that
  close. A miss also costs clue 6 and ~14 s, because the clue 6 trip then starts from the
  dead-reckoned fallback and usually misses. A likely fix is riding ~0.1 m further from the
  south wall until clue 5 is read.
- **Rare snag at the pond's edge** (true x ≈ 1.25, y ≈ −2.2) while stepping out for clue 6:
  one watchdog recovery in 15 runs (~8 s lost).
- The wall rides themselves (~55 s) are now the biggest phase. The next candidates there are a
  quicker speed response (`speed_kp`), a wider on-line tolerance, and faster corner turns.
