Mordred
=================

Mordred is a research prototype of a heterogeneous CPU-GPU DBMS engine.
The work was presented at VLDB '22. Please read the [paper](https://dl.acm.org/doi/abs/10.14778/3551793.3551809) for more details. 

```
@article{10.14778/3551793.3551809,
author = {Yogatama, Bobbi W. and Gong, Weiwei and Yu, Xiangyao},
title = {Orchestrating Data Placement and Query Execution in Heterogeneous CPU-GPU DBMS},
year = {2022},
issue_date = {July 2022},
publisher = {VLDB Endowment},
volume = {15},
number = {11},
issn = {2150-8097},
url = {https://doi.org/10.14778/3551793.3551809},
doi = {10.14778/3551793.3551809},
journal = {Proc. VLDB Endow.},
month = {jul},
pages = {2491–2503},
numpages = {13}
}
```

**Mordred is currently under development. The repository is now moved to https://github.com/bwyogatama/Mordred**

**Reach out to bwyogatama@cs.wisc.edu for questions or issues.**

Usage
----

*Dependencies:
```
Ubuntu > 20.04
GCC > 9.4
CUDA > 11.0
IntelTBB
```

To use Mordred:

To run the Star Schema Benchmark implementation:

* Generate the test dataset

```
cd test/

# Generate the test generator / transformer binaries
cd ssb/dbgen
make
cd ../loader
make 
cd ../../

# Generate the test data and transform into columnar layout
# Substitute <SF> with appropriate scale factor (eg: 1)
python util.py ssb <SF> gen
python util.py ssb <SF> transform

#Sort the LINEORDER table if you want to enable segment skipping
cd ssb/loader
make sort
./columnSort ../data/s{SF}_columnar/LINEORDER ../data/s{SF}_columnar/LINEORDERSORT 5 16 {COLUMN_SIZE}
```

* Configure the benchmark settings
```
# Edit SF and BASE_PATH in src/ssb/ssb_utils.h
# Edit SF and BASE_PATH in src/ssb/common.h
# Edit file path in src/ssb/minmax.cpp and src/ssb/minmaxsort.cpp
# Edit SF and table size in minmax.sh
```

* To compile and run Mordred
```
make setup
make bin/gpudb/minmax
make bin/gpudb/minmaxsort
./minmax.sh
make bin/gpudb/main
./bin/gpudb/main
```
