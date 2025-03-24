import pandas as pd

# Excel file paths
features_file = r' C:path\to\files\Features_Progresion.xlsx'
tics_file = r' C:path\to\files\TICs_Normalizadas.xlsx'
output_file = r' C:path\to\files\Features_Progresion_with_TIC.xlsx'

# Read the Excel files
df_features = pd.read_excel(features_file)
df_tics = pd.read_excel(tics_file)

# Ensure that the identification column has the same name in both DataFrames.
# If the column in df_tics is 'Paciente', rename it to 'PatientID'
df_tics.rename(columns={'Paciente': 'PatientID'}, inplace=True)

# Transform df_tics from wide to long format.
# Include 'rCBV' and 'PSR' in id_vars to keep them in the DataFrame.
df_tics_long = df_tics.melt(
    id_vars=['PatientID', 'rCBV', 'PSR'],
    var_name='Timepoint',
    value_name='TIC'
)

# Extract the Timepoint number from the 'Timepoint' column (which has values like 'TIC_Normalizada_1', 'TIC_Normalizada_2', etc.)
df_tics_long['Timepoint'] = df_tics_long['Timepoint'].str.replace('TIC_Normalizada_', '').astype(int)

# Merge the DataFrames based on 'PatientID' and 'Timepoint'
df_merged = pd.merge(df_features, df_tics_long, on=['PatientID', 'Timepoint'], how='left')

# Save the resulting DataFrame to a new Excel file
df_merged.to_excel(output_file, index=False)
print(f"File saved at: {output_file}")
