# Sudoku Grid Extraction

This project focuses on extracting and processing Sudoku grids from images using classical computer vision techniques.
It includes functionalities for detecting the grid, warping the image, and preparing it for further analysis or digit recognition.

![Sudoku Extraction](https://github.com/user-attachments/assets/5f763479-8b19-4e54-bf05-fb9ab1c76a44)

## Quickstart

This project supports `uv` for easy environment management and script execution.
To run the extraction, you can use the provided script.
Some sample sudoku images are available in the `images/` directory.

```bash
uv run scripts/extraction.py --image images/sudoku_001.jpg
```

<img width="640" alt="image" src="https://github.com/user-attachments/assets/1d401a42-a33a-47d4-9054-02344e530f07" />

<img width="480" alt="image" src="https://github.com/user-attachments/assets/86cf2168-055d-4791-9af3-6bb7dd2c4441" />


### Corner Sorting

The project includes a utility for sorting the corners of the detected Sudoku grid in a consistent order (top-left, top-right, bottom-right, bottom-left).

```bash
uv run scripts/visualize_corner_sorting.py --type simple
```

```bash
uv run scripts/visualize_corner_sorting.py --help
usage: visualize_corner_sorting.py [-h] [--type {cyclic,assignment,centroid,simple}] [--fps FPS] [--rot_speed ROT_SPEED]
```

## Processing Pipeline

The main processing steps include:
1. **Grayscale Conversion**: Convert the input image to grayscale.
2. **Canny Edge Detection**: Apply Canny edge detection to highlight the edges in the image.
```bash
uv run scripts/visualize_corner_sorting.py --help
usage: visualize_corner_sorting.py [-h] [--type {cyclic,assignment,centroid,simple}] [--fps FPS] [--rot_speed ROT_SPEED]
```

## Processing Pipeline

The main processing steps include:
1. **Grayscale Conversion**: Convert the input image to grayscale.
2. **Canny Edge Detection**: Apply Canny edge detection to highlight the edges in the image.
3. **Contour Detection**: Find contours in the edge-detected image.
4. **Grid Detection**: Identify the largest quadrilateral contour, which is assumed to be the Sudoku grid.
5. **Corner Sorting**: Sort the corners of the detected grid using a specified method (e.g., cyclic sort with centroid anchor).
6. **Perspective Transformation**: Warp the image to obtain a top-down view of the Sudoku grid.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [SciPy Linear Sum Assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)

