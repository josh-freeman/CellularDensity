import numpy as np
from utils import get_map_white_pixels_to_respresentatives

def test_get_map_white_pixels_to_respresentatives():
    # Create a test mask with two distinct white regions
    test_mask = [[0, 0, 0, 0, 0, 0, 0],
                 [255, 255, 255, 255, 0, 0, 0],
                 [0, 255, 255, 0, 0, 0, 0],
                 [0, 255, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 255, 255],
                 [0, 0, 0, 0, 0, 255, 255]]
    
    test_mask = np.array(test_mask, dtype=np.uint8)

    # Call the function
    result = get_map_white_pixels_to_respresentatives(test_mask)

    # Print the result
    print(f"Found {len(result)} white pixel groups:")
    for group in result:
        print(group)
    
    # Assertions to validate the result
    assert len(result) == 2, "There should be two groups of white pixels"


# Run the test
test_get_map_white_pixels_to_respresentatives()