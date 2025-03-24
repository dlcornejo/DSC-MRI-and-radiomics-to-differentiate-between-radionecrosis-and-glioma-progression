import os
import SimpleITK as sitk
import logging
import numpy as np
import subprocess
import pandas as pd

# General configuration
input_dir = r"C:path\to\files"  # Main input directory
excel_path = os.path.join(input_dir, 'metricas_pacientes.xlsx')  # Path for the output Excel file
dcm2niix_executable = 'dcm2niix'  # dcm2niix executable
output_filename_prefix = 'Perfusion'  # Output filename prefix for perfusion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize a global error counter
error_count = 0

# Function to obtain detailed metrics of an image
def get_image_metrics(image):
    # Convert the image to a NumPy array to compute metrics
    image_array = sitk.GetArrayFromImage(image)
    dims = image_array.shape  # Image dimensions
    spacing = image.GetSpacing()  # Spacing between image pixels
    direction = image.GetDirection()  # Image direction in space
    origin = image.GetOrigin()  # Image origin in space
    mean = np.mean(image_array)  # Mean of pixel values
    min_val = np.min(image_array)  # Minimum pixel value
    max_val = np.max(image_array)  # Maximum pixel value
    var_val = np.var(image_array)  # Variance of pixel values
    std_dev = np.std(image_array)  # Standard deviation of pixel values
    return dims, spacing, direction, origin, mean, min_val, max_val, var_val, std_dev

# Function to convert NRRD to NIfTI while preserving the pixel type
def convert_nrrd_to_nifti(input_path, output_path, convert_to_float32=True):
    global error_count
    logger.info(f"Convirtiendo {input_path} a NIfTI en {output_path}")
    try:
        # Read the NRRD image from the file
        image = sitk.ReadImage(input_path)
        original_metrics = get_image_metrics(image)  # Obtain metrics from the original image
    except Exception as e:
        logger.error(f"Error al leer {input_path}: {e}")
        error_count += 1
        return None, None, None

    # Get the image array
    image_array = sitk.GetArrayFromImage(image)
    # If necessary, convert to float32
    if convert_to_float32:
        image_array = image_array.astype('float32')
    else:
        # Preserve the original data type
        pixel_type = image.GetPixelIDTypeAsString()
        logger.info(f"Preservando el tipo de píxel: {pixel_type}")
        # Convert the NumPy array to the corresponding type
        if pixel_type == '8-bit unsigned integer':
            image_array = image_array.astype(np.uint8)
        elif pixel_type == '16-bit unsigned integer':
            image_array = image_array.astype(np.uint16)
        elif pixel_type == '32-bit signed integer':
            image_array = image_array.astype(np.int32)
        elif pixel_type == '64-bit signed integer':
            image_array = image_array.astype(np.int64)
        elif pixel_type == '32-bit float':
            image_array = image_array.astype(np.float32)
        elif pixel_type == '64-bit float':
            image_array = image_array.astype(np.float64)
        else:
            logger.warning(f"Tipo de píxel desconocido: {pixel_type}. Se mantendrá el tipo de dato actual.")

    # **New: Convert non-zero values to 1 to binarize the image**
    image_array = (image_array != 0).astype(image_array.dtype)

    # Create a new image from the array and set spatial properties
    new_image = sitk.GetImageFromArray(image_array)
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(image.GetOrigin())

    try:
        # Write the new image to the output file
        sitk.WriteImage(new_image, output_path)
        if not os.path.exists(output_path):
            logger.error(f"Error al escribir {output_path}: El archivo no se creó correctamente.")
            error_count += 1
            return None, None, None
    except Exception as e:
        logger.error(f"Error al escribir {output_path}: {e}")
        error_count += 1
        return None, None, None

    logger.info(f"Conversión completada para {input_path}")
    # Read the NIfTI image to obtain its metrics
    nifti_image = sitk.ReadImage(output_path)
    nifti_metrics = get_image_metrics(nifti_image)
    logger.info(f"Tipo de píxel de la imagen convertida: {nifti_image.GetPixelIDTypeAsString()}")
    return original_metrics, nifti_metrics, nifti_image

# Function to convert DICOM to NIfTI using dcm2niix
def convert_dicom_to_nifti(dicom_dir, output_nifti_path, output_json_path):
    try:
        logger.info(f"Convirtiendo DICOM en {dicom_dir} a NIfTI {output_nifti_path}")
        # Execute dcm2niix to convert the DICOM file to NIfTI
        subprocess.run([
            dcm2niix_executable,
            '-f', output_filename_prefix,  # Output filename
            '-m', 'y',  # Group multiple time files
            '-p', 'y',  # Allow corrections for Philips
            '-z', 'y',  # Compress output to .nii.gz
            '-ba', 'y',  # Include JSON file in the output
            '-i', 'y',  # Do not invert axes
            '-o', dicom_dir, dicom_dir
        ], check=True)

        # Search for the generated NIfTI files
        nifti_file = None
        json_file = None
        for file in os.listdir(dicom_dir):
            if file.endswith(".nii.gz") and output_filename_prefix in file:
                nifti_file = os.path.join(dicom_dir, file)
            if file.endswith(".json") and output_filename_prefix in file:
                json_file = os.path.join(dicom_dir, file)

        # Move the generated files to the specified output location
        if nifti_file:
            if os.path.exists(output_nifti_path):
                os.remove(output_nifti_path)
            os.rename(nifti_file, output_nifti_path)
            logger.info(f"Archivo de perfusion guardado como {output_nifti_path}")
        else:
            logger.error(f"No se encontro un archivo NIfTI generado en {dicom_dir}.")

        if json_file:
            if os.path.exists(output_json_path):
                os.remove(output_json_path)
            os.rename(json_file, output_json_path)
            logger.info(f"Archivo JSON de perfusion guardado como {output_json_path}")

        # Read the NIfTI image to obtain its metrics
        nifti_image = sitk.ReadImage(output_nifti_path)
        nifti_metrics = get_image_metrics(nifti_image)
        return nifti_metrics, nifti_image
    except Exception as e:
        logger.error(f"Error al convertir los DICOM en {dicom_dir}: {e}")
        return None, None

# Function to align perfusion (4D) with segmentation (3D)
def align_perfusion_to_segmentation(perf_image, seg_image):
    logger.info("Alineando la perfusion con la segmentacion.")
    # Get spatial properties of the segmentation
    seg_spacing = seg_image.GetSpacing()    # Segmentation spacing
    seg_direction = seg_image.GetDirection()  # Segmentation direction
    seg_origin = seg_image.GetOrigin()        # Segmentation origin
    seg_size = seg_image.GetSize()            # Segmentation size

    # Create a list to store the resampled perfusion volumes
    resampled_volumes = []

    # Iterate over each temporal acquisition of the perfusion (4D) to align it with the segmentation (3D)
    for t in range(perf_image.GetSize()[3]):
        logger.info(f"Remuestreando la perfusion para la adquisicion temporal {t+1} de {perf_image.GetSize()[3]}.")
        # Extract the 3D volume corresponding to temporal acquisition t
        extractor_index = [0, 0, 0, t]
        size = list(perf_image.GetSize()[:3]) + [0]  # The 0 indicates that only a single image is extracted in the temporal dimension
        perf_volume = sitk.Extract(perf_image, size, extractor_index)

        # Configure the resampling filter to align perfusion with segmentation
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)       # Linear interpolation
        resample.SetOutputSpacing(seg_spacing)          # Set output spacing to that of the segmentation
        resample.SetSize(seg_size)                      # Set output size to that of the segmentation
        resample.SetOutputDirection(seg_direction)      # Set output direction to that of the segmentation
        resample.SetOutputOrigin(seg_origin)            # Set output origin to that of the segmentation
        resample.SetDefaultPixelValue(0)                # Default value for pixels out of range

        # Resample the perfusion so that it matches the segmentation
        resampled_perf_image = resample.Execute(perf_volume)
        # Convert the resampled volume to an array and add it to the list
        resampled_volumes.append(sitk.GetArrayFromImage(resampled_perf_image))

    # Combine the resampled volumes into a 4D image (keeping the temporal axis intact)
    resampled_perf_4d = np.stack(resampled_volumes, axis=-1)
    # Convert the resampled 4D image to SimpleITK format
    resampled_perf_image_4d = sitk.GetImageFromArray(resampled_perf_4d)
    resampled_perf_image_4d.SetSpacing(seg_spacing)
    resampled_perf_image_4d.SetDirection(seg_direction)
    resampled_perf_image_4d.SetOrigin(seg_origin)
    return resampled_perf_image_4d

# Function to process images and store the results in an Excel file
def process_images_and_save_metrics(input_dir, excel_path):
    patient_data = []  # List to store data for each patient
    for patient_folder in os.listdir(input_dir):
        patient_dir = os.path.join(input_dir, patient_folder)
        # Check if the directory corresponds to a patient
        if os.path.isdir(patient_dir):
            logger.info(f"Procesando paciente {patient_folder}")
            # Define paths for segmentation, NAWM, and perfusion files
            segmentation_file = os.path.join(patient_dir, 'Segmentation.seg.nrrd')
            nawm_file = os.path.join(patient_dir, 'NAWM.seg.nrrd')  # New file to process
            perfusion_dir = os.path.join(patient_dir, 'perfusion')

            # Define output paths for NIfTI files
            nifti_segmentation = os.path.join(patient_dir, 'Segmentation.nii.gz')
            nifti_nawm = os.path.join(patient_dir, 'NAWM.nii.gz')  # Output path for NAWM
            nifti_perfusion = os.path.join(patient_dir, 'Perfusion.nii.gz')
            json_perfusion = os.path.join(patient_dir, 'Perfusion.json')

            # Convert NRRD segmentation to NIfTI without converting to float32
            if os.path.exists(segmentation_file):
                seg_original_metrics, seg_converted_metrics, seg_image = convert_nrrd_to_nifti(
                    segmentation_file, nifti_segmentation, convert_to_float32=False
                )
            else:
                logger.warning(f"Segmentacion no encontrada para el paciente {patient_folder}.")
                continue

            # Convert NAWM.seg.nrrd to NIfTI without converting to float32
            if os.path.exists(nawm_file):
                nawm_original_metrics, nawm_converted_metrics, nawm_image = convert_nrrd_to_nifti(
                    nawm_file, nifti_nawm, convert_to_float32=False
                )
            else:
                logger.warning(f"NAWM.seg.nrrd no encontrado para el paciente {patient_folder}.")
                nawm_original_metrics = nawm_converted_metrics = nawm_image = None

            # Convert DICOM perfusion to NIfTI
            if os.path.exists(perfusion_dir):
                perf_metrics, perf_image = convert_dicom_to_nifti(perfusion_dir, nifti_perfusion, json_perfusion)
                if perf_metrics is None or perf_image is None:
                    logger.error(f"Error en la conversion de perfusion para el paciente {patient_folder}")
                    continue
            else:
                logger.warning(f"Perfusion no encontrada para el paciente {patient_folder}.")
                continue

            # Align perfusion with segmentation using the actual images
            aligned_perf_image = align_perfusion_to_segmentation(perf_image, seg_image)
            # Save the aligned perfusion
            aligned_perf_output_path = os.path.join(patient_dir, 'Aligned_Perfusion.nii.gz')
            sitk.WriteImage(aligned_perf_image, aligned_perf_output_path)
            logger.info(f"Perfusion alineada guardada en {aligned_perf_output_path}")

            # Prepare the metrics dictionary for this patient
            patient_metrics = {
                'Paciente': patient_folder,
                'Dim_Segmentacion_Original': seg_original_metrics[0],
                'Spacing_Segmentacion_Original': seg_original_metrics[1],
                'Direction_Segmentacion_Original': seg_original_metrics[2],
                'Origin_Segmentacion_Original': seg_original_metrics[3],
                'Mean_Segmentacion_Original': seg_original_metrics[4],
                'Min_Segmentacion_Original': seg_original_metrics[5],
                'Max_Segmentacion_Original': seg_original_metrics[6],
                'Var_Segmentacion_Original': seg_original_metrics[7],
                'StdDev_Segmentacion_Original': seg_original_metrics[8],
                'Dim_Segmentacion_Convertida': seg_converted_metrics[0],
                'Spacing_Segmentacion_Convertida': seg_converted_metrics[1],
                'Direction_Segmentacion_Convertida': seg_converted_metrics[2],
                'Origin_Segmentacion_Convertida': seg_converted_metrics[3],
                'Mean_Segmentacion_Convertida': seg_converted_metrics[4],
                'Min_Segmentacion_Convertida': seg_converted_metrics[5],
                'Max_Segmentacion_Convertida': seg_converted_metrics[6],
                'Var_Segmentacion_Convertida': seg_converted_metrics[7],
                'StdDev_Segmentacion_Convertida': seg_converted_metrics[8],
                'Dim_Perfusion_Original': perf_metrics[0],
                'Spacing_Perfusion_Original': perf_metrics[1],
                'Direction_Perfusion_Original': perf_metrics[2],
                'Origin_Perfusion_Original': perf_metrics[3],
                'Mean_Perfusion_Original': perf_metrics[4],
                'Min_Perfusion_Original': perf_metrics[5],
                'Max_Perfusion_Original': perf_metrics[6],
                'Var_Perfusion_Original': perf_metrics[7],
                'StdDev_Perfusion_Original': perf_metrics[8]
            }

            # Add NAWM metrics if available
            if nawm_original_metrics is not None and nawm_converted_metrics is not None:
                patient_metrics.update({
                    'Dim_NAWM_Original': nawm_original_metrics[0],
                    'Spacing_NAWM_Original': nawm_original_metrics[1],
                    'Direction_NAWM_Original': nawm_original_metrics[2],
                    'Origin_NAWM_Original': nawm_original_metrics[3],
                    'Mean_NAWM_Original': nawm_original_metrics[4],
                    'Min_NAWM_Original': nawm_original_metrics[5],
                    'Max_NAWM_Original': nawm_original_metrics[6],
                    'Var_NAWM_Original': nawm_original_metrics[7],
                    'StdDev_NAWM_Original': nawm_original_metrics[8],
                    'Dim_NAWM_Convertida': nawm_converted_metrics[0],
                    'Spacing_NAWM_Convertida': nawm_converted_metrics[1],
                    'Direction_NAWM_Convertida': nawm_converted_metrics[2],
                    'Origin_NAWM_Convertida': nawm_converted_metrics[3],
                    'Mean_NAWM_Convertida': nawm_converted_metrics[4],
                    'Min_NAWM_Convertida': nawm_converted_metrics[5],
                    'Max_NAWM_Convertida': nawm_converted_metrics[6],
                    'Var_NAWM_Convertida': nawm_converted_metrics[7],
                    'StdDev_NAWM_Convertida': nawm_converted_metrics[8]
                })
            else:
                # If NAWM was not found, fill with NaN values
                nawm_keys = [
                    'Dim_NAWM_Original', 'Spacing_NAWM_Original', 'Direction_NAWM_Original',
                    'Origin_NAWM_Original', 'Mean_NAWM_Original', 'Min_NAWM_Original',
                    'Max_NAWM_Original', 'Var_NAWM_Original', 'StdDev_NAWM_Original',
                    'Dim_NAWM_Convertida', 'Spacing_NAWM_Convertida', 'Direction_NAWM_Convertida',
                    'Origin_NAWM_Convertida', 'Mean_NAWM_Convertida', 'Min_NAWM_Convertida',
                    'Max_NAWM_Convertida', 'Var_NAWM_Convertida', 'StdDev_NAWM_Convertida'
                ]
                for key in nawm_keys:
                    patient_metrics[key] = np.nan

            # Add patient metrics to the list
            patient_data.append(patient_metrics)

    # Create a DataFrame with patient information
    df = pd.DataFrame(patient_data)
    # Save the information to an Excel file
    df.to_excel(excel_path, index=False)
    logger.info(f"Métricas guardadas en {excel_path}")

# Run the process
process_images_and_save_metrics(input_dir, excel_path)
