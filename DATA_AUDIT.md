# Data audit

**Source:** [UCI Parking Birmingham](https://archive.ics.uci.edu/dataset/482/parking+birmingham), downloaded locally from the supplied CSV on 2026-10-04. Attribution: UCI Machine Learning Repository, Parking Birmingham dataset. Check the official page for the current licence terms.

The supplied file has 35,717 rows, 30 car-park codes, and four expected columns. `Occupancy` is an integer count of occupied spaces: values are on the same scale as `Capacity`, and `Capacity - Occupancy` is a meaningful availability calculation. It is not a rate.

Observed timestamps span `2016-10-04 07:46:28` to `2016-12-19 16:30:35` and are interpreted as Birmingham local time, then localised with the `Europe/London` timezone rules. The median within-car-park interval is about 32.9 minutes, but gaps range from zero to several days. Daytime coverage is visible in the roughly 07:46–16:30 range; this must not be represented as a current live service.

## Cleaning decisions

The pipeline records a SHA-256 checksum in `artifacts/manifest.json`. It removes exact duplicates, unparsable rows, non-positive capacities, negative occupancy, and occupancy above capacity. The initial audit found 216 exact duplicate rows, 0 missing values, 0 non-positive capacities, 12 negative occupancy values, and 373 over-capacity values. Excluded rows are recorded in the manifest. No weather, events, coordinates, or names are invented.

Targets use an exact same-car-park match at `t + 30 minutes`. A following row is never treated as a target. Exact matching is intentionally not replaced by interpolation; the manifest records the observed coverage. The current supplied file produces sparse exact labels, so production work should first consider a documented tolerance and report the actual horizon distribution.
