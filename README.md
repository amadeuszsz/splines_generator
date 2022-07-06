# Splines Generator
Simple splines generator. Application allows to draw a splines with a given image as a background and save all 
consecutive points to json file. Output file contains list with size (m, n, d), where
* m - num of splines,
* n - num of points for a single spline (default 128, use `--points` arg to modify)
* d - spline dimension, equals 2, (x, y) order

## Installation
```
git clone https://github.com/amadeuszsz/splines_generator
cd splines_generator
pip3 install -r requirements.txt
```

## Usage
* Check possible args
```
python3 splines_generator.py --help 
```

* Run splines generator
```
python3 splines_generator --filepath /path/to/image/file.jpg 
```
See command output for app instruction.
