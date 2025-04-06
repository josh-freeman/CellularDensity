# CellularDensity
Some hyperparameters:
- Kernel size
- Percentile of Red intensity value
- Number of kernel iterations
- size of kernel when doing adaptive thresholding 

Further stuff:
- Heatmap implementation
1. Get the centers of each zone in one set of points, `representatives`.
2. Pick a height and width granularity (TODO later there is probably a way to have a more optimized variable rectangle size depending on how similar values are)
3. Dividing the image into a grid of w*h rectangles, count how many `representatives` each grid rectangle contains as a fraction of the total. Color it accordingly.
- Histogram stretching
- OTSU formula
