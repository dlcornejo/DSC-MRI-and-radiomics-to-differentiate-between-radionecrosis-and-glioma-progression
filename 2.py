import os
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import multiprocessing

# ---------------------------
# Configuration Parameters
# ---------------------------
# Directory where the patient folders are located 
base_dir = r' C:path\to\files'
print(f"Base directory: {base_dir}")

# Output directory
output_dir = r' C:path\to\files '

# Parameter to control the number of parallel processes
num_processes = 16  # Adjust according to system capacity

# Log level (change to logging.DEBUG for more details)
log_level = logging.INFO

# Name of the log file
log_filename = os.path.join(output_dir, 'log_extraccionP.txt')

# Name of the image file for feature extraction
image_name = 'Aligned_Perfusion.nii.gz'

# Name of the output file with the extracted features
output_filename = 'Features.xlsx'

# PyRadiomics configuration parameters
params = {
    'setting': {
        'binWidth': 5,
        'normalize': True,
        'normalizeScale': 100,
        'resampledPixelSpacing': [1.75, 1.75, 4],
        'interpolator': 'sitkBSpline',
        'padDistance': 5,
        'geometryTolerance': 1e-3,
        'preCrop': True,
        'label': 1,
        'additionalInfo': True,
    },
    'imageType': {
        'Original': {},
        'Wavelet': {}
    },
    'featureClass': {
        'firstorder': [],
        'glcm': [],
        'gldm': [],
        'glrlm': [],
        'glszm': [],
        'ngtdm': [],  # Add third-order features (NGTDM)
        'shape': []    # Add shape features
    }
}

# ---------------------------
# Logger Configuration
# ---------------------------
# Create the output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Configure the logger to save to file and display in terminal
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure the PyRadiomics logger
import radiomics
radiomics.logger.setLevel(log_level)
logger.info(f'The results folder has been verified/created at: {output_dir}')

# ---------------------------
# Initialize the Feature Extractor
# ---------------------------
# Initialize the feature extractor by passing 'params'
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# ---------------------------
# Main Functions
# ---------------------------
# Function to extract features from an image
def extract_features(patient_id, image_sitk, mask_sitk, timepoint=None):
    if timepoint is not None:
        logger.info(f'Extracting features for patient {patient_id}, time sequence {timepoint + 1}...')
    else:
        logger.info(f'Extracting features for patient {patient_id}...')
    
    # Crop the image and mask to the tumor bounding box
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask_sitk)
    bounding_box = label_shape_filter.GetBoundingBox(1)  # Assuming label 1 for the tumor
    index = bounding_box[:3]
    size = bounding_box[3:]
    
    # Verify that the bounding box has a valid size
    if any(s == 0 for s in size):
        logger.error(f'Invalid bounding box for patient {patient_id}, timepoint {timepoint + 1 if timepoint is not None else 1}')
        return None

    # Extract the region of interest from the image and the mask
    cropped_image = sitk.RegionOfInterest(image_sitk, size, index)
    cropped_mask = sitk.RegionOfInterest(mask_sitk, size, index)
    # Ensure that the cropped mask has the same metadata as the cropped image
    cropped_mask.CopyInformation(cropped_image)
    
    # Check if the mask has at least one significant voxel
    if np.sum(sitk.GetArrayFromImage(cropped_mask)) == 0:
        logger.error(f'The mask is empty for patient {patient_id}, timepoint {timepoint + 1 if timepoint is not None else 1}')
        return None

    # Check if the image has vector pixels
    if cropped_image.GetNumberOfComponentsPerPixel() > 1:
        # Select the desired component (e.g., the first component)
        cropped_image = sitk.VectorIndexSelectionCast(cropped_image, 0)
    
    # Check intensities within the mask
    masked_array = sitk.GetArrayFromImage(cropped_image)[sitk.GetArrayFromImage(cropped_mask) > 0]
    min_intensity = masked_array.min()
    max_intensity = masked_array.max()
    logger.info(f"Patient {patient_id}, Timepoint {timepoint + 1 if timepoint is not None else 1}: Minimum intensity = {min_intensity}, Maximum intensity = {max_intensity}")
    
    if np.isnan(masked_array).any() or np.isinf(masked_array).any():
        logger.error(f'NaN or Inf values found in the image for patient {patient_id}, timepoint {timepoint + 1 if timepoint is not None else 1}')
        return None

    try:
        # Execute feature extraction using the previously configured extractor
        result = extractor.execute(cropped_image, cropped_mask)
    except Exception as e:
        logger.error(f'Error extracting features for patient {patient_id}, timepoint {timepoint + 1 if timepoint is not None else 1}: {e}')
        return None

    # Add metadata
    result_data = dict(result)
    result_data['PatientID'] = patient_id
    result_data['Timepoint'] = timepoint + 1 if timepoint is not None else 1  # Ensure 'Timepoint' is always present
    return result_data

def process_patient(args):
    patient_id, image_name = args
    output_data = []
    try:
        patient_dir = os.path.join(base_dir, patient_id)
        logger.info(f'Processing patient: {patient_id}')
        # Paths for the image and segmentation files
        image_file = os.path.join(patient_dir, image_name)
        segmentation_file = os.path.join(patient_dir, 'Segmentation.nii.gz')
        
        if os.path.exists(image_file) and os.path.exists(segmentation_file):
            logger.info(f'Files {image_name} and segmentation found for patient {patient_id}')
            # Read the images using SimpleITK
            image_sitk = sitk.ReadImage(image_file)
            segmentation_mask_sitk = sitk.ReadImage(segmentation_file)
            
            # Verify that the spatial dimensions match
            if image_sitk.GetSize()[:3] != segmentation_mask_sitk.GetSize():
                logger.warning(f'Image and mask dimensions do not match for patient {patient_id}')
                return None

            # Get the number of components per pixel (for images with vector pixels)
            num_components = image_sitk.GetNumberOfComponentsPerPixel()
            num_dimensions = image_sitk.GetDimension()

            if num_components > 1:
                logger.info(f'The image has vector pixels with {num_components} components')
                for timepoint in range(num_components):
                    # Select the component corresponding to the timepoint
                    image_3d_sitk = sitk.VectorIndexSelectionCast(image_sitk, timepoint)
                    result_data = extract_features(patient_id, image_3d_sitk, segmentation_mask_sitk, timepoint)
                    if result_data is not None:
                        output_data.append(result_data)
                    else:
                        logger.warning(f'Features skipped for patient {patient_id}, timepoint {timepoint + 1} due to an error.')
            elif num_dimensions == 4:
                num_timepoints = image_sitk.GetSize()[3]
                for timepoint in range(num_timepoints):
                    # Extract the 3D image corresponding to the timepoint
                    extractor_index = [0, 0, 0, timepoint]
                    size = list(image_sitk.GetSize()[:3]) + [0]
                    image_3d_sitk = sitk.Extract(image_sitk, size, extractor_index)
                    result_data = extract_features(patient_id, image_3d_sitk, segmentation_mask_sitk, timepoint)
                    if result_data is not None:
                        output_data.append(result_data)
                    else:
                        logger.warning(f'Features skipped for patient {patient_id}, timepoint {timepoint + 1} due to an error.')
            elif num_dimensions == 3:
                # The image is 3D, process directly
                result_data = extract_features(patient_id, image_sitk, segmentation_mask_sitk)
                if result_data is not None:
                    output_data.append(result_data)
                else:
                    logger.warning(f'Features skipped for patient {patient_id} due to an error.')
            else:
                logger.warning(f'The image for patient {patient_id} is neither 3D nor 4D. Dimensions: {image_sitk.GetSize()}')
                return None
        else:
            logger.warning(f'Files {image_name} or segmentation not found for patient {patient_id}')
            return None
    except Exception as e:
        logger.error(f'Error processing patient {patient_id}: {e}')
        return None
    return output_data

def extract_features_for_image(image_name, output_filename):
    output_data = []
    # Get the list of patients
    patient_ids = os.listdir(base_dir)
    # Prepare arguments for multiprocessing
    args_list = [(patient_id, image_name) for patient_id in patient_ids]
    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the process_patient function in parallel
        results = list(tqdm(pool.imap(process_patient, args_list),
                            total=len(args_list),
                            desc=f"Processing patients for {image_name}"))
    # Collect the results
    for result in results:
        if result is not None:
            output_data.extend(result)
    # Create a DataFrame with all results
    df = pd.DataFrame(output_data)
    if not df.empty:
        # Reorder columns so that 'PatientID' and 'Timepoint' come first
        cols = ['PatientID', 'Timepoint'] + [col for col in df.columns if col not in ['PatientID', 'Timepoint']]
        df = df[cols]
        # Save the results to an Excel file
        output_file = os.path.join(output_dir, output_filename)
        df.to_excel(output_file, index=False)
        logger.info(f'Feature extraction completed for {image_name}. File saved at: {output_file}')
    else:
        logger.error('No features were extracted. The DataFrame is empty.')

# ---------------------------
# Script Execution
# ---------------------------
if __name__ == '__main__':
    # Call the main function with the image and output file names
    extract_features_for_image(image_name, output_filename)
