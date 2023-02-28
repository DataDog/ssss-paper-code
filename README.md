# SpaceSavingSets

Heavy Distinct Hitter sketch implementations in Rust.

Code used to generate data for the Experiments section of:

Homin K. Lee and Charles Masson.
Sampling Space-Saving Set Sketches.

## Usage
```
const SSSS_SIZE: usize = 10;
const SEEDS: [u64; 4] = [0, 1, 2, 3];
const CARDINALITY_SKETCH_SIZE: usize = 256;
const HLL_SEEDS: [u64; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

let config = ssss::Config::new(
    SSSS_SIZE,
    hll::Config::new(CARDINALITY_SKETCH_SIZE, Some(HLL_SEEDS)).unwrap(),
    Some(SEEDS),
).unwrap();

let mut sketch: ssss::HllSamplingSpaceSavingSets<u64, u64> =
    ssss::SamplingSpaceSavingSets::new(&config);

for i in 1..11 {
    let label = i * 10;
    for item in 0..label {
        sketch.insert(label, &item);
    }
}
println!("{:?}", sketch.top(2));
```

## Requirements
The code is written in [Rust](https://www.rust-lang.org/).

## Compile
```
cargo build --release
```

## Running on Example Data
```
./target/release/benchmarks combo benchmarks/data/example -v -m 0.5
```
