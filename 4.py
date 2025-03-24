import pandas as pd

# Load the data from the previously generated Excel file
input_excel = r"C:path\to\files"
df = pd.read_excel(input_excel)

# Drop all columns that contain the word "diagnostics" or whose name is "Unnamed"
columns_to_drop = [col for col in df.columns if 'diagnostics' in col or 'Unnamed' in col]
df_cleaned = df.drop(columns=columns_to_drop)

# Reorder the columns 
df_cleaned = df_cleaned.reset_index(drop=True)

# Save the cleaned DataFrame to a new Excel file
output_excel = r" C:path\to\files\Features.xlsx"
df_cleaned.to_excel(output_excel, index=False)
print(f"Cleaned data saved to {output_excel}")
