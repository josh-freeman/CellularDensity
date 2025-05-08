import os
#INPUT_PATH = "dataset/sample_tiles/imgs/"
#OUTPUT_PATH = "dataset/sample_tiles/pred/"
# Constants
KERNEL_SIZE = (5, 5)
DILATION_ITERATIONS = 2
RPB_THRESHOLD_PERCENTILE = 50
ALPHA_OVERLAY = 0.5
GRABCUT_ITERATIONS = 5
OUTPUT_TILES_PATH = os.path.expanduser("~/Downloads/ndpi Files analysis")
INPUT_TILES_PATH = os.path.expanduser("~/Downloads/ndpi_files")
OUTPUT_STATS_FILENAME = "output_stats.json"
TILE_SIZE = 512
TOP_BIGGEST_CONTOURS_TO_OBSERVE = 5
CONTOUR_LEVEL = 4
MIN_FRACTION_OF_TILE_INSIDE_CONTOUR = 0.5