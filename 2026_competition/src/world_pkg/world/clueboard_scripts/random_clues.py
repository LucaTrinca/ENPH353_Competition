"""Test only: write a clue file with random values, for the clue board generator.

    python3 random_clues.py            # writes random_clues.csv next to this script
    python3 random_clues.py --seed 4   # the same random clues every time

Then start the simulation with the generator reading it instead of the course URL:

    CLUES_URL=~/2026_competition_ws/src/2026_competition/src/world_pkg/world/clueboard_scripts/random_clues.csv \
        ros2 launch my_robot_bringup my_robot.launch.py

The file has the course URL's format: the 8 clue types on the first line, their
values on the second. The types stay the standard ones (the score tracker and
the boards' order depend on them); the values are random: 1 - 12 characters
from A-Z and 0-9, with up to 2 single spaces between words.
"""
import argparse
import random
from pathlib import Path

TYPES = ['SIZE', 'VICTIM', 'CRIME', 'TIME', 'PLACE', 'MOTIVE', 'WEAPON', 'BANDIT']
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'
MAX_LENGTH = 12


def random_word(length):
    # Mostly letters, sometimes digits (real clues have both, e.g. LAP66, 0003)
    digit_share = random.choice([0.0, 0.0, 0.2, 0.5, 1.0])
    return ''.join(random.choice(DIGITS if random.random() < digit_share else LETTERS)
                   for _ in range(length))


def random_value():
    length = random.randint(1, MAX_LENGTH)
    spaces = random.randint(0, min(2, (length - 1) // 2))   # words at least 1 character
    # Split the non-space characters into spaces + 1 words
    letters = length - spaces
    cuts = sorted(random.sample(range(1, letters), spaces)) if spaces else []
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [letters])]
    return ' '.join(random_word(n) for n in sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'random_clues.csv')
    args = parser.parse_args()
    random.seed(args.seed)

    values = [random_value() for _ in TYPES]
    args.out.write_text(','.join(TYPES) + '\n' + ','.join(values) + '\n')
    print(f'Wrote {args.out}')
    for key, value in zip(TYPES, values):
        print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
