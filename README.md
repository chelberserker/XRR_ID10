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

Clone the repository to your working directory and import the `XRR` class in your python script.

```bash
git clone https://github.com/chelberserker/XRR_ID10
```

## Usage

Here is a basic example of how to use the library:

```python
from XRR import XRR
import matplotlib.pyplot as plt

# Define parameters
file_path = 'path/to/your/data.h5'
zgH_ScanN_list = [21] # zgH scan before XRR scan to normalize the intensity, optional
refl_ScanN_list = [22,23] # List of reflectivity scan numbers to process

# Initialize the XRR object
# This automatically loads data and processes it
xrr = XRR(file=file_path, scans=refl_ScanN_list)
zgH = XRR(file=file_path, scans=zgH_ScanN_list)

# Apply all corrections: transmission, flux, footprint
xrr.apply_auto_corrections(sample_size=17, beam_size=19, z_scan=zgH)

# Plot reflectivity
xrr.plot_reflectivity()

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

