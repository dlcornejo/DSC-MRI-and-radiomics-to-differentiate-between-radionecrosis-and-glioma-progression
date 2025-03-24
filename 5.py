import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import json

# Logging configuration
logging.basicConfig(
    filename=r' C:ruta\a\archivos\diagnostico_TICs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================
# Configurable Parameters
# =============================
directorio_pacientes = r" C:path\to\files"
directorio_resultados = r" C:path\to\files"
num_points = 40  
interpolation_method = 'cubic'  # Interpolation method: 'linear' or 'cubic'
calcular_rcbv_psr = True  # Calculate rCBV and PSR
sigma_tumor_normalizada = 1  # Sigma value to smooth the tumor's normalized signal

# Ensure that the results directory exists
os.makedirs(directorio_resultados, exist_ok=True)

# Function to load NIfTI images using SimpleITK
def cargar_nifti_sitk(file_path):
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)

# Function to load TR (Repetition Time) from a JSON file
def cargar_tr_desde_json(json_file_path):
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    if 'RepetitionTime' in json_data:
        tr = json_data['RepetitionTime']
        logger.info(f"Repetition Time (TR) extracted from JSON: {tr}")
        return tr
    else:
        raise KeyError("Field 'RepetitionTime' not found in the JSON.")

# Function to calculate acquisition times
def calcular_tiempos_adquisicion(tr, num_frames):
    return np.array([tr * i for i in range(num_frames)])

# Function to extract the average TIC from a segmented area
def extraer_tic_segmentada(perfusion_img, segmentation_mask):
    num_frames = perfusion_img.shape[-1]
    tic_values = []
    for frame in range(num_frames):
        # Apply the mask to each temporal frame
        frame_data = perfusion_img[..., frame]
        masked_data = frame_data[segmentation_mask > 0]
        if masked_data.size > 0:
            tic_values.append(np.mean(masked_data))
        else:
            tic_values.append(np.nan)
    return np.array(tic_values)

# Function to calculate TTA (Time To Arrival)  approach V2
def calcular_tta(tic_nawm, tiempos, sigma=0.001, slope_threshold=-1):
    # Smooth the signal to reduce noise
    tic_nawm_smooth = gaussian_filter1d(tic_nawm, sigma=sigma)
    # Calculate the slope between consecutive points
    slopes = np.diff(tic_nawm_smooth) / np.diff(tiempos)
    # Find the index of the MSID (minimum of the signal)
    msid_idx = np.argmin(tic_nawm_smooth)
    # Initialize tta_index as None
    tta_index = None
    # Iterate backwards from msid_idx
    for i in range(msid_idx - 1, 0, -1):
        # If the slope is greater than the threshold (stops decreasing)
        if slopes[i - 1] > slope_threshold:
            tta_index = i
            break
    if tta_index is None:
        logger.warning("TTA could not be detected. Skipping normalization.")
        return None, tic_nawm_smooth
    return tta_index, tic_nawm_smooth

# Function to calculate rCBV and PSR
def calcular_rcbv_psr_func(tiempo_normalizado, tic_normalizada, ttp_index):
    # Calculate rCBV as the area under the curve from TTP to the end
    rCBV = np.trapz(tic_normalizada[ttp_index:], tiempo_normalizado[ttp_index:])
    # Calculate PSR as the percentage of signal recovery
    psr = (tic_normalizada[-1] - tic_normalizada[ttp_index]) / (tic_normalizada[0] - tic_normalizada[ttp_index]) * 100
    return rCBV, psr

# Initialize a dictionary to store the results
resultados_tic = {"Paciente": []}

# Process each folder in the patients directory
for paciente_id in os.listdir(directorio_pacientes):
    carpeta_paciente = os.path.join(directorio_pacientes, paciente_id)
    # Ensure it is a directory and not a file
    if not os.path.isdir(carpeta_paciente):
        logger.info(f"Ignoring unrelated file: {paciente_id}")
        continue
    logger.info(f"Processing patient: {paciente_id}")
    try:
        # Check if the necessary files exist
        perfusion_path = os.path.join(carpeta_paciente, "Aligned_Perfusion.nii.gz")
        tumor_segmentation_path = os.path.join(carpeta_paciente, "Segmentation.nii.gz")
        nawm_segmentation_path = os.path.join(carpeta_paciente, "NAWM.nii.gz")
        json_perf_path = os.path.join(carpeta_paciente, "Perfusion.json")  # JSON file with TR
        if (os.path.exists(perfusion_path) and os.path.exists(tumor_segmentation_path) and 
            os.path.exists(nawm_segmentation_path) and os.path.exists(json_perf_path)):
            logger.info(f"Files found for patient {paciente_id}")
            # Load TR from the JSON file
            tr = cargar_tr_desde_json(json_perf_path)
            # Load images using SimpleITK
            img_perf = cargar_nifti_sitk(perfusion_path)  # Dimensions: (X, Y, Z, T)
            img_tumor_seg = cargar_nifti_sitk(tumor_segmentation_path)  # Dimensions: (X, Y, Z)
            img_nawm_seg = cargar_nifti_sitk(nawm_segmentation_path)  # Dimensions: (X, Y, Z)
            # Ensure that spatial dimensions match
            assert img_perf.shape[:3] == img_tumor_seg.shape == img_nawm_seg.shape, \
                f"Spatial dimensions do not match for patient {paciente_id}"
            # Calculate acquisition times
            num_frames = img_perf.shape[-1]
            tiempos = calcular_tiempos_adquisicion(tr, num_frames)
            # Extract the TICs
            tic_tumor = extraer_tic_segmentada(img_perf, img_tumor_seg)
            tic_nawm = extraer_tic_segmentada(img_perf, img_nawm_seg)
            # Check if the TICs have valid values
            if np.all(np.isnan(tic_tumor)) or np.all(np.isnan(tic_nawm)):
                logger.warning(f"Empty TICs for patient {paciente_id}. Skipping.")
                continue
            # Detect TTA using the calcular_tta function
            tta_nawm_index, tic_nawm_smooth = calcular_tta(tic_nawm, tiempos)
            if tta_nawm_index is None:
                continue  # A warning was already logged in calcular_tta
            # Detect TTP (MSID)
            ttp_nawm_index = np.argmin(tic_nawm_smooth)
            ttp_tta_nawm = tiempos[ttp_nawm_index] - tiempos[tta_nawm_index]
            if ttp_tta_nawm <= 0:
                logger.warning(f"TTP - TTA is not positive for patient {paciente_id}. Skipping normalization.")
                continue
            # Align the baseline and the initial point of the descending curve
            baseline_nawm = np.mean(tic_nawm[:tta_nawm_index])
            baseline_tumor = np.mean(tic_tumor[:tta_nawm_index])
            tic_nawm_aligned = tic_nawm - baseline_nawm
            tic_tumor_aligned = tic_tumor - baseline_tumor
            shift_nawm = tic_nawm_aligned[tta_nawm_index]
            shift_tumor = tic_tumor_aligned[tta_nawm_index]
            tic_nawm_aligned -= shift_nawm
            tic_tumor_aligned -= shift_tumor
            # Ensure that intensities are positive
            min_intensity = min(np.min(tic_nawm_aligned), np.min(tic_tumor_aligned))
            if min_intensity < 0:
                tic_nawm_aligned -= min_intensity
                tic_tumor_aligned -= min_intensity
            # Calculate MSID_NAWM
            msid_nawm = np.max(tic_nawm_aligned) - np.min(tic_nawm_aligned)
            if msid_nawm <= 0:
                logger.warning(f"MSID_NAWM is not positive for patient {paciente_id}. Skipping normalization.")
                continue
            # Normalize the tumor TIC
            tic_tumor_normalizada = tic_tumor_aligned / msid_nawm
            # **Apply smoothing to the tumor's normalized signal**
            tic_tumor_normalizada_suavizada = gaussian_filter1d(tic_tumor_normalizada, sigma=sigma_tumor_normalizada)
            # Normalize the times
            tiempo_normalizado = (tiempos - tiempos[tta_nawm_index]) / ttp_tta_nawm
            # Limit the time points to the range of valid data
            valid_indices = (tiempo_normalizado >= tiempo_normalizado[0]) & (tiempo_normalizado <= tiempo_normalizado[-1])
            tiempo_normalizado = tiempo_normalizado[valid_indices]
            tic_tumor_normalizada_suavizada = tic_tumor_normalizada_suavizada[valid_indices]
            # Interpolate to constant time points within the valid range
            normalized_time_points = np.linspace(tiempo_normalizado[0], tiempo_normalizado[-1], num_points)
            # Interpolate
            interpolator = interp1d(
                tiempo_normalizado,
                tic_tumor_normalizada_suavizada,
                kind=interpolation_method,
                bounds_error=False,
                fill_value="extrapolate"
            )
            tic_tumor_interpolated = interpolator(normalized_time_points)
            # Save the results in the dictionary
            resultados_tic["Paciente"].append(paciente_id)
            # Save the normalized TIC values
            for i, valor in enumerate(tic_tumor_interpolated):
                columna = f"TIC_Normalizada_{i + 1}"
                if columna not in resultados_tic:
                    resultados_tic[columna] = []
                resultados_tic[columna].append(valor)
            # If requested, calculate rCBV and PSR
            if calcular_rcbv_psr:
                # Find the TTP index in the normalized time
                ttp_normalizado = tiempo_normalizado[ttp_nawm_index]
                ttp_index = np.searchsorted(normalized_time_points, ttp_normalizado)
                # Ensure the index is within limits
                ttp_index = min(max(ttp_index, 0), len(normalized_time_points) - 1)
                rCBV, psr = calcular_rcbv_psr_func(normalized_time_points, tic_tumor_interpolated, ttp_index)
                # Save rCBV and PSR in the results
                resultados_tic.setdefault("rCBV", []).append(rCBV)
                resultados_tic.setdefault("PSR", []).append(psr)
            # Generate plots
            plt.figure(figsize=(20, 8))
            # Plot of aligned TICs
            plt.subplot(1, 3, 1)
            plt.plot(tiempos, tic_tumor_aligned, label='TIC Tumor (aligned)', color='blue')
            plt.plot(tiempos, tic_nawm_aligned, label='TIC NAWM (aligned)', color='orange')
            plt.axvline(x=tiempos[tta_nawm_index], color='green', linestyle='--', label='TTA NAWM')
            plt.axvline(x=tiempos[ttp_nawm_index], color='red', linestyle='--', label='TTP NAWM (MSID)')
            plt.title('Aligned TICs')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal Intensity (aligned)')
            plt.legend()
            # Plot of NAWM TIC with TTA and TTP
            plt.subplot(1, 3, 2)
            plt.plot(tiempos, tic_nawm_aligned, label='TIC NAWM (aligned)', color='orange')
            plt.axvline(x=tiempos[tta_nawm_index], color='green', linestyle='--', label='TTA NAWM')
            plt.axvline(x=tiempos[ttp_nawm_index], color='red', linestyle='--', label='TTP NAWM (MSID)')
            plt.scatter(tiempos[tta_nawm_index], tic_nawm_aligned[tta_nawm_index], color='green', zorder=5)
            plt.scatter(tiempos[ttp_nawm_index], tic_nawm_aligned[ttp_nawm_index], color='red', zorder=5)
            plt.title('NAWM TIC with TTA and TTP')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal Intensity (aligned)')
            plt.legend()
            # Plot of normalized and interpolated Tumor TIC
            plt.subplot(1, 3, 3)
            plt.plot(normalized_time_points, tic_tumor_interpolated, label='Normalized Tumor TIC', color='red')
            plt.title('Normalized, Smoothed and Interpolated Tumor TIC')
            plt.xlabel('Normalized Time (units of TTP - TTA)')
            plt.ylabel('Normalized Signal Intensity')
            plt.legend()
            # If rCBV and PSR were calculated, display them in the title
            if calcular_rcbv_psr:
                plt.suptitle(f'Patient {paciente_id} | rCBV: {rCBV:.2f}, PSR: {psr:.2f}%')
            else:
                plt.suptitle(f'Patient {paciente_id}')
            plt.tight_layout()
            # Save the plot
            grafico_path = os.path.join(directorio_resultados, f'TIC_{paciente_id}.png')
            plt.savefig(grafico_path)
            plt.close()
            logger.info(f"Processing completed for patient {paciente_id}")
        else:
            logger.warning(f"Necessary files not found for patient {paciente_id}")
            continue  # Move to the next patient
    except Exception as e:
        logger.error(f"Error processing patient {paciente_id}: {e}")

# Create a DataFrame with the results
df_resultados = pd.DataFrame(resultados_tic)
# Save the results in an Excel file
output_path = os.path.join(directorio_resultados, "TICs_Normalizadas.xlsx")
df_resultados.to_excel(output_path, index=False)
logger.info(f"Results saved at: {output_path}")
print(f"Results saved at: {output_path}")
