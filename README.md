Mordred
=================

Usage
----

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
```

* Configure the benchmark settings
```
cd src/ssb/
# Edit SF and BASE_PATH in ssb_utils.h
```

* To compile and run Mordred
```
make setup
make bin/gpudb/main
./bin/gpudb/main
```