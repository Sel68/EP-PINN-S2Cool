import pandas as pd
import sys
import os

def convert_xlsx_to_csv(file_path):
    # Read all sheets to include all data
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    
    # Combine all sheets into a single DataFrame
    combined_data = pd.concat(all_sheets.values(), ignore_index=True)
    
    # Create the exact same filename with a .csv extension
    csv_path = os.path.splitext(file_path)[0] + '.csv'
    
    # Export to CSV
    combined_data.to_csv(csv_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename.xlsx>")
        sys.exit(1)
        
    convert_xlsx_to_csv(sys.argv[1])