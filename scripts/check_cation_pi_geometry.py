#!/usr/bin/env python3
"""Does PandaMap's cation-pi label correspond to actual stacking geometry?

It mostly does not, and this matters before the mode-switch result is written up.
Of 133 arginine guanidinium groups within 8 A of an aromatic amine ring centroid
(Rank 0 models, the five aromatic neutral amines), only 14% sit within 6 A AND
within 30 degrees of the ring normal -- textbook cation-pi. The median is 5.54 A
at 50 degrees off-normal, i.e. edge-on proximity rather than a stack.

Genuine stacks do exist: 19 cases across 11 of 26 enzymes, and 14 of the 19 are
at position 222 rather than 206. So the claim a proposal can carry is that the
arginine stays engaged with aromatic amines through a non-ionic mode, with a
minority showing true cation-pi geometry -- not that the arginines stack.
"""
