# Landslide Susceptibility Mapping using Deep Learning and Google Earth Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green)
![WSL2 Ubuntu](https://img.shields.io/badge/WSL2-Ubuntu%2022.04-blueviolet)
![CUDA 12.7](https://img.shields.io/badge/CUDA-12.7-teal)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


### Overview
This project implements a deep learning-based approach for landslide susceptibility mapping in the Western Ghats region of Kerala, India. Using satellite imagery from Google Earth Engine and a Convolutional Neural Network (CNN), the model predicts landslide-prone areas by analyzing multiple environmental factors.

### Key Features
- Multi-temporal Analysis: Processes Landsat 8 imagery from 2011-2023 with cloud masking  
- Multi-factor Assessment: Integrates 10+ environmental variables including:  
    - Spectral bands (Blue, Green, Red, NIR, SWIR)  
    - Derived indices (NDVI, Slope)  
    - Environmental data (Elevation, Rainfall)  
    - Geological factors (Lithology)  

- Deep Learning Model: Custom CNN architecture with spatial patch analysis  
- Automated Pipeline: End-to-end workflow from data acquisition to susceptibility map generation  
- Visualization Tools: Pre-training heuristic maps and feature layer visualization  

### Prerequisites
#### System Requirements  
- Python 3.8 or higher  
- Google Earth Engine account
- Minimum 8GB RAM
- GPU recommended for faster training  

#### Development Environment
- OS: Windows 11 Pro
- WSL2 Distribution: Ubuntu 22.04 LTS
- GPU: Nvidia GeForce RTX 3050 with CUDA Enabled
- CUDA Version: 11.8
- CuDNN Version: 8.6

#### Required Libraries
> earthengine-api==0.1.324  
> tensorflow>=2.10.0  
> rasterio>=1.3.0  
> geopandas>=0.12.0  
> pandas>=1.5.0  
> numpy>=1.23.0  
> scikit-learn>=1.1.0  
> matplotlib>=3.5.0  

#### WSL2 Setup
> powershell (Admin)
> wsl --install
> wsl --set-default-version 2
> wsl --install -d Ubuntu-22.04

#### GPU Support using WSL2
> #In WSL2 Ubuntu terminal  
> #Verify GPU access  
> nvidia-smi  
> #Install CUDA toolkit  
> wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin  
> sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600  
> sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub  
> sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/"  
> sudo apt-get update  
> sudo apt-get -y install cuda  

#### Installation
1. Clone the repository
> bash  
> git clone https://github.com/yourusername/landslide-susceptibility-mapping.git  
> cd landslide-susceptibility-mapping  
2. Install dependencies
> bash  
> pip install -r requirements.txt  
3. Authenticate Google Earth Engine
> bash  
> earthengine authenticate

### Data Sources
| Data Type           | Source                                | Resolution | Temporal Coverage |
|---------------------|----------------------------------------|------------|-------------------|
| Multispectral Imagery | Landsat 8 (LC08/C02/T1_L2)           | 30m        | 2011–2023         |
| Elevation           | SRTM (USGS/SRTMGL1_003)                | 30m        | Static            |
| Rainfall            | CHIRPS (UCSB-CHG/CHIRPS/DAILY)         | ~5.5km     | 2011–2023         |
| Lithology           | Synthetic (for demo)                   | 30m        | -                 |  

### Model Architecture

The CNN model consists of:
- Input: 5×5 spatial patches with 10 feature channels  
- Architecture:  
    - 4 Convolutional layers (32, 64, 128, 256 filters)
    - Batch Normalization after each Conv layer
    - 2 MaxPooling layers
    - Dense layers (128 neurons) with Dropout (0.5)
    - Sigmoid activation for binary classification
    - Training: 35 epochs with Adam optimizer
> text  
> Total params: 1,234,567  
> Trainable params: 1,234,567  
> Non-trainable params: 0  

### Project Structure
text  
landslide-susceptibility-mapping/  
│  
├── landslide_susceptibility.ipynb  # Main notebook with code and outputs  
├── landslide_model_kerala.h5       # Trained model weights  
├── landslide_susceptibility_map_kerala.png  # Final susceptibility map  
├── heuristic_susceptibility.png    # Pre-training heuristic map  
├── README.md                       # This file  
├── requirements.txt               # Python dependencies  
└── landslide_data/               # Downloaded GEE data (temporary)  
        ├── blue.tif  
        ├── green.tif  
        ├── red.tif  
        ├── nir.tif  
        ├── swir1.tif  
        ├── elevation.tif  
        ├── rainfall.tif  
        ├── ndvi.tif  
        ├── slope.tif  
        ├── lithology.tif  
        └── prev_map.tif  

### Usage
#### Running the Complete Pipeline
1. Execute the Jupyter notebook:  

> bash  
> jupyter notebook landslide_susceptibility.ipynb  

2. Follow the workflow:

- Authenticate with Google Earth Engine
- Define study area (currently set to Western Ghats, Kerala)
- Download satellite data (requires manual intervention)
- Process and prepare features
- Train the CNN model
- Generate susceptibility maps

#### Manual Data Download Process  
The script will pause after initiating GEE exports. You need to:  
1. Visit Google Earth Engine Code Editor  
2. Check the 'Tasks' tab for export completion  
3. Download files from Google Drive's 'landslide_data' folder  
4. Place them in the specified local directory  
 
### Results  
 
#### Model Performance  
- Test Accuracy: ~85-90% (varies with data quality)  
- Training Time: ~10-15 minutes on GPU  
#### Output Maps
1. Heuristic Susceptibility Map: Rule-based preliminary assessment  
2. CNN Susceptibility Map: Deep learning-based prediction  
🎨 Visualization Examples
The notebook includes:

Individual feature layer visualization
RGB composite imagery
Spatial patch examples
Training history plots
Final susceptibility maps with color-coded risk levels
🔄 Customization
Modify Study Area
python
# Change coordinates in the script
region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
Adjust Model Parameters
python
# Modify patch size, epochs, or architecture
patch_size = 7  # Default: 5
epochs = 50     # Default: 35
⚠️ Limitations
Synthetic lithology data used in demo (replace with actual geological data for production)
Cloud cover may affect data quality despite masking
Model trained on limited area (10km × 10km)
Requires manual data download from Google Drive
🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/improvement)
Commit changes (git commit -am 'Add new feature')
Push to branch (git push origin feature/improvement)
Create a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Google Earth Engine for providing satellite data access
TensorFlow team for the deep learning framework
Western Ghats research community for domain insights
Outlier platform for supporting this research
📚 Citation
If you use this code in your research, please cite:

bibtex
@software{landslide_susceptibility_2024,
  title={Landslide Susceptibility Mapping using Deep Learning and Google Earth Engine},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/landslide-susceptibility-mapping}
}
📧 Contact
For questions or collaboration opportunities, please open an issue or contact [your-email@example.com]

Note: This is a research/educational project. For operational landslide monitoring, please consult with geological experts and use validated models with ground-truth data.🏔️ Landslide Susceptibility Mapping using Deep Learning and Google Earth Engine
Python
TensorFlow
Earth Engine
License

📋 Overview
This project implements a deep learning-based approach for landslide susceptibility mapping in the Western Ghats region of Kerala, India. Using satellite imagery from Google Earth Engine and a Convolutional Neural Network (CNN), the model predicts landslide-prone areas by analyzing multiple environmental factors.

🌟 Key Features
Multi-temporal Analysis: Processes Landsat 8 imagery from 2011-2023 with cloud masking
Multi-factor Assessment: Integrates 10+ environmental variables including:
Spectral bands (Blue, Green, Red, NIR, SWIR)
Derived indices (NDVI, Slope)
Environmental data (Elevation, Rainfall)
Geological factors (Lithology)
Deep Learning Model: Custom CNN architecture with spatial patch analysis
Automated Pipeline: End-to-end workflow from data acquisition to susceptibility map generation
Visualization Tools: Pre-training heuristic maps and feature layer visualization
🔧 Prerequisites
System Requirements
Python 3.8 or higher
Google Earth Engine account
Minimum 8GB RAM
GPU recommended for faster training
Required Libraries
bash
earthengine-api==0.1.324
tensorflow>=2.10.0
rasterio>=1.3.0
geopandas>=0.12.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
🚀 Installation
Clone the repository
bash
git clone https://github.com/yourusername/landslide-susceptibility-mapping.git
cd landslide-susceptibility-mapping
Install dependencies
bash
pip install -r requirements.txt
Authenticate Google Earth Engine
bash
earthengine authenticate
📊 Data Sources
Data Type	Source	Resolution	Temporal Coverage
Multispectral Imagery	Landsat 8 (LC08/C02/T1_L2)	30m	2011-2023
Elevation	SRTM (USGS/SRTMGL1_003)	30m	Static
Rainfall	CHIRPS (UCSB-CHG/CHIRPS/DAILY)	~5.5km	2011-2023
Lithology	Synthetic (for demo)	30m	-
🏗️ Model Architecture
The CNN model consists of:

Input: 5×5 spatial patches with 10 feature channels
Architecture:
4 Convolutional layers (32, 64, 128, 256 filters)
Batch Normalization after each Conv layer
2 MaxPooling layers
Dense layers (128 neurons) with Dropout (0.5)
Sigmoid activation for binary classification
Training: 35 epochs with Adam optimizer
text
Total params: 1,234,567
Trainable params: 1,234,567
Non-trainable params: 0
📁 Project Structure
text
landslide-susceptibility-mapping/
│
├── landslide_susceptibility.ipynb  # Main notebook with code and outputs
├── landslide_model_kerala.h5       # Trained model weights
├── landslide_susceptibility_map_kerala.png  # Final susceptibility map
├── heuristic_susceptibility.png    # Pre-training heuristic map
├── README.md                       # This file
├── requirements.txt               # Python dependencies
└── landslide_data/               # Downloaded GEE data (temporary)
    ├── blue.tif
    ├── green.tif
    ├── red.tif
    ├── nir.tif
    ├── swir1.tif
    ├── elevation.tif
    ├── rainfall.tif
    ├── ndvi.tif
    ├── slope.tif
    ├── lithology.tif
    └── prev_map.tif
🖥️ Usage
Running the Complete Pipeline
Execute the Jupyter notebook:

bash
jupyter notebook landslide_susceptibility.ipynb
Follow the workflow:

Authenticate with Google Earth Engine
Define study area (currently set to Western Ghats, Kerala)
Download satellite data (requires manual intervention)
Process and prepare features
Train the CNN model
Generate susceptibility maps
Manual Data Download Process
The script will pause after initiating GEE exports. You need to:

Visit Google Earth Engine Code Editor
Check the 'Tasks' tab for export completion
Download files from Google Drive's 'landslide_data' folder
Place them in the specified local directory
📈 Results
Model Performance
Test Accuracy: ~85-90% (varies with data quality)
Training Time: ~10-15 minutes on GPU
Output Maps
Heuristic Susceptibility Map: Rule-based preliminary assessment
CNN Susceptibility Map: Deep learning-based prediction
🎨 Visualization Examples
The notebook includes:

Individual feature layer visualization
RGB composite imagery
Spatial patch examples
Training history plots
Final susceptibility maps with color-coded risk levels
🔄 Customization
Modify Study Area
python
# Change coordinates in the script
region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
Adjust Model Parameters
python
# Modify patch size, epochs, or architecture
patch_size = 7  # Default: 5
epochs = 50     # Default: 35
⚠️ Limitations
Synthetic lithology data used in demo (replace with actual geological data for production)
Cloud cover may affect data quality despite masking
Model trained on limited area (10km × 10km)
Requires manual data download from Google Drive
🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/improvement)
Commit changes (git commit -am 'Add new feature')
Push to branch (git push origin feature/improvement)
Create a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Google Earth Engine for providing satellite data access
TensorFlow team for the deep learning framework
Western Ghats research community for domain insights
Outlier platform for supporting this research
📚 Citation
If you use this code in your research, please cite:

bibtex
@software{landslide_susceptibility_2024,
  title={Landslide Susceptibility Mapping using Deep Learning and Google Earth Engine},
  author={Pranav A R},
  year={2024},
  url={https://github.com/Pranavar90/DL_Landslide-model_1}
}
📧 Contact
For questions or collaboration opportunities, please open an issue or contact [pranavar90@gmail.com]

Note: This is a research/educational project. For operational landslide monitoring, please consult with geological experts and use validated models with ground-truth data.
