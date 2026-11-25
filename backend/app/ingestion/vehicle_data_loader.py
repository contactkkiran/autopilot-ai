from pathlib import Path
import pandas as pd

# --------------------------------------------------
# STEP 1: Locate the project root directory
# --------------------------------------------------

# __file__ gives the location of the current Python file.
# resolve() converts it into an absolute path.
# parents[3] moves three directory levels up
# to locate the project root.

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------
# STEP 2: Locate the synthetic vehicle dataset folder
# --------------------------------------------------

# Build the complete path to the folder containing CSV files.

DATASET_DIR = PROJECT_ROOT / "datasets" / "raw" / "kit"


# --------------------------------------------------
# STEP 3: Load vehicle sensor data
# --------------------------------------------------


def load_vehicle_data():

    try:

        # Find all CSV files available in the dataset folder.
        # sorted() arranges filenames alphabetically.

        csv_files = sorted(DATASET_DIR.glob("*.csv"))

        # Check whether any CSV files are available.

        if not csv_files:
            raise FileNotFoundError("No vehicle CSV files found")

        # Select the first CSV file.

        file_path = csv_files[0]

        # Display the name of the CSV file being loaded.

        print(f"Loading file: {file_path.name}")

        # Read the CSV file into a Pandas DataFrame.

        df = pd.read_csv(file_path)

        # --------------------------------------------------
        # STEP 4: Display dataset information
        # --------------------------------------------------

        print("\n--- Dataset Information ---")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        # --------------------------------------------------
        # STEP 5: Display all column names
        # --------------------------------------------------

        print("\n--- Column Names ---")
        print(df.columns.tolist())

        # --------------------------------------------------
        # STEP 6: Display first 5 CSV records
        # --------------------------------------------------

        print("\n--- First 5 CSV Records ---")

        with pd.option_context("display.max_columns", None, "display.width", 300):
            print(df.head(5))

        # --------------------------------------------------
        # STEP 7: Analyse missing data
        # --------------------------------------------------

        # Count missing values in each column.

        print("\n--- Missing Values Per Column ---")

        missing_values = df.isnull().sum()
        print(missing_values)

        # Calculate missing-value percentage per column.

        print("\n--- Missing Data Percentage ---")

        missing_percentage = (df.isnull().sum() / len(df)) * 100
        print(missing_percentage.round(2))

        # Calculate total missing cells.

        total_missing = df.isnull().sum().sum()

        print("\n--- Total Missing Values ---")
        print(total_missing)

        # --------------------------------------------------
        # STEP 8: Analyse duplicate records
        # --------------------------------------------------

        # Identify duplicate rows in the DataFrame.

        duplicate_records = df.duplicated()

        # Count total duplicate records.

        total_duplicates = duplicate_records.sum()

        print("\n--- Duplicate Records Analysis ---")

        if total_duplicates == 0:
            print("No duplicate rows found")
        else:
            print(f"Total Duplicate Records: {total_duplicates}")

        # --------------------------------------------------
        # STEP 9: Return the loaded DataFrame
        # --------------------------------------------------

        return df

    # --------------------------------------------------
    # STEP 10: Handle exceptions
    # --------------------------------------------------

    except FileNotFoundError as error:
        print(f"File Error: {error}")

    except pd.errors.EmptyDataError:
        print("CSV Error: The CSV file is empty.")

    except pd.errors.ParserError as error:
        print(f"CSV Parsing Error: {error}")

    except PermissionError as error:
        print(f"Permission Error: {error}")

    except Exception as error:
        print(f"Unexpected Error: {error}")

    return None


# --------------------------------------------------
# STEP 11: Execute the data loader
# --------------------------------------------------

if __name__ == "__main__":
    vehicle_df = load_vehicle_data()
