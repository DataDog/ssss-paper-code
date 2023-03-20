<meta name="robots" content="noindex">

# IMPORTANT

*IMPORTANT*

Reading the LICENSE.md file will invalidate double-blind anonymity.

# SpaceSavingSets

Heavy Distinct Hitter sketch implementations in Rust.

Code used to generate data for the Experiments section of:

Sampling Space-Saving Set Sketches.

## Usage

```rs
use rand::{rngs::StdRng, Rng, SeedableRng};
use sketch_traits::{HeavyDistinctHitterSketch, New};

let mut seeded_rng = StdRng::seed_from_u64(12345);

const SSSS_SIZE: usize = 10;
const CARDINALITY_SKETCH_SIZE: usize = 256;

let config = ssss::Config::new(
    SSSS_SIZE,
    hll::Config::new(CARDINALITY_SKETCH_SIZE, seeded_rng.gen()).unwrap(),
    seeded_rng.gen(),
)
.unwrap();

let mut sketch = ssss::HllSamplingSpaceSavingSets::new(&config);

for label in (10..=100).step_by(10) {
    for item in 0..label {
        sketch.insert(label, &item);
    }
}

assert_eq!(sketch.top(2), [(&100, 101), (&90, 87)]);
```

## Requirements

The code is written in [Rust](https://www.rust-lang.org/).

## Compiling

```
cargo build --release
```

## Running the benchmarks on Example Data

```
cargo run --release -- combo benchmarks/data/example -v -m 0.5
```
