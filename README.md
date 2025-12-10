# XRR_ID10_ESRF

This library provides tools for processing Surface X-ray Scattering data obtained on the ID10 beamline at ESRF.

## Features

*   **Data Loading**: Efficiently loads HDF5 files produced at ID10.
*   **Data Processing**: Performs region-of-interest (ROI) based integration of 2D detector data.
*   **Corrections**: Applies necessary corrections including:
    *   Conversion to $q_z$
    *   Error calculation
    *   Footprint correction
    *   Transmission correction
    *   Rebinning
*   **Reciprocal Space Mapping**: Calculates and plots reciprocal space maps ($q_x$, $q_z$).
*   **Visualization**: Plotting and saving of reflectivity curves and detector images.

## Installation

### Dependencies

*   Python 3.x
*   h5py
*   matplotlib
*   numpy
*   scipy

You can install the required packages using pip:

```bash
pip install h5py matplotlib numpy scipy
```

### Setup

Clone the repository and import the `XRR` class in your python script.

```bash
git clone https://github.com/your_username/XRR_ID10_ESRF.git
```

## Usage

Here is a basic example of how to use the library:

```python
from XRR import XRR
import matplotlib.pyplot as plt

# Define parameters
file_path = 'path/to/your/data.h5'
scan_numbers = [10, 11, 12]  # List of scan numbers to process

# Initialize the XRR object
# This automatically loads data and processes it
xrr = XRR(file=file_path, scans=scan_numbers)

# Plot reflectivity
fig, ax = xrr.plot_reflectivity()
plt.show()

# Apply footprint correction
xrr.footprint_correction(sample_size=1.0, beamsize=100)

# Re-plot to see corrected data
xrr.plot_reflectivity()
plt.show()

# Save the reflectivity data
xrr.save_reflectivity()
```

## Documentation

Full documentation is available in the `docs/_build/html` directory. You can view it by opening `docs/_build/html/index.html` in your web browser.

To regenerate the documentation (requires `sphinx` and `sphinx-rtd-theme`):

```bash
cd docs
make html
```

## Contributing

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-branch`).
3.  Make your changes.
4.  Submit a pull request.

