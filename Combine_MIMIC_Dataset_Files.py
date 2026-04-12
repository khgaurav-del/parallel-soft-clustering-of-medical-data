# This program aggregates multiple MIMIC-III dataset files into a single comprehensive dataset,
# using Polars for efficient data manipulation. 
# It processes the ADMISSIONS, DIAGNOSES_ICD, D_ICD_DIAGNOSES, PRESCRIPTIONS, LABEVENTS, MICROBIOLOGYEVENTS,
# and PROCEDUREEVENTS_MV tables, performing necessary joins and aggregations to create a unified dataset that can be used for analysis or modeling.
# The final output is saved in both CSV and Parquet formats for flexibility in downstream applications.

#? We use Polars for its speed and efficiency, especially when working with large datasets like MIMIC-III. 
#? Pandas can struggle with memory and performance when handling large files, while Polars is designed to be more efficient in these scenarios.

import polars as pl
import time

def aggregate_all_mimic_data(output_path):
    print("Starting complete parallel aggregation...")
    start_time = time.time()

    base_dir = r"MIMIC-III Dataset\MIMIC -III (10000 patients)"
    
    standard_overrides = {"HADM_ID": pl.Float64}
    diag_overrides = {"HADM_ID": pl.Float64, "ICD9_CODE": pl.String}
    dict_overrides = {"ICD9_CODE": pl.String}

    try:  #try block to catch any unexpected errors during processing 
       
        # 1. Base Table # Aka Admissions - we start with this as our base since it contains the primary key (HADM_ID) that we will join all other tables on.
        admissions = pl.scan_csv(
            f"{base_dir}\\ADMISSIONS\\ADMISSIOMS_sorted_cleaned.csv",
            schema_overrides=standard_overrides # We override HADM_ID to Float64 to handle any potential nulls or inconsistencies, and we will cast it to Int64 after filtering out nulls
        ).with_columns(pl.col("HADM_ID").cast(pl.Int64))

        # 2. Diagnoses 
        print("Processing Diagnoses...")
        diag_events = pl.scan_csv(
            f"{base_dir}\\DIAGNOSES_ICD\\DIAGNOSES_ICD_sorted.csv",
            schema_overrides=diag_overrides
        ).filter(pl.col("HADM_ID").is_not_null()).with_columns(pl.col("HADM_ID").cast(pl.Int64))
        
        diag_dict = pl.scan_csv(
            f"{base_dir}\\D_ICD_DIAGNOSES\\D_ICD_DIAGNOSES.csv",
            schema_overrides=dict_overrides
        )
        
        diagnoses = (
            diag_events.join(diag_dict, on="ICD9_CODE", how="left")
            .group_by("HADM_ID").agg([
                pl.col("LONG_TITLE").drop_nulls().alias("DIAGNOSIS_NAMES"),
                pl.col("ICD9_CODE").count().alias("TOTAL_DIAGNOSES")
            ])
        )

        # 3. Prescriptions
        print("Processing Prescriptions...")
        prescriptions = pl.scan_csv(
            f"{base_dir}\\PRESCRIPTIONS\\PRESCRIPTIONS_grouped.csv",
            schema_overrides=standard_overrides
        ).filter(pl.col("HADM_ID").is_not_null()).with_columns(pl.col("HADM_ID").cast(pl.Int64)).group_by("HADM_ID").agg([
            pl.col("DRUG").drop_nulls().alias("PRESCRIBED_DRUGS"),
            pl.col("DRUG").count().alias("TOTAL_PRESCRIPTIONS")
        ])

        # 4. Lab Events
        print("Processing Lab Events...")
        lab_events = pl.scan_csv(
            f"{base_dir}\\LABEVENTS\\LABEVENTS_sorted.csv",
            schema_overrides=standard_overrides
        ).filter(pl.col("HADM_ID").is_not_null()).with_columns(pl.col("HADM_ID").cast(pl.Int64)).group_by("HADM_ID").agg([
            pl.col("ITEMID").count().alias("TOTAL_LAB_EVENTS")
        ])

        # 5. Microbiology Events
        print("Processing Microbiology...")
        microbiology = pl.scan_csv(
            f"{base_dir}\\MICROBIOLOGYEVENTS\\MICROBIOLOGYEVENTS_sorted.csv",
            schema_overrides=standard_overrides
        ).filter(pl.col("HADM_ID").is_not_null()).with_columns(pl.col("HADM_ID").cast(pl.Int64)).group_by("HADM_ID").agg([
            pl.col("ORG_NAME").drop_nulls().unique().alias("MICROBIOLOGY_ORGANISMS") 
        ])

        # 6. Procedure Events MV
        print("Processing Mechanical Ventilation Procedures...")
        procedures_mv = pl.scan_csv(
            f"{base_dir}\\PROCEDUREEVENTS_MV\\PROCEDUREEVENTS_MV_sorted.csv",
            schema_overrides=standard_overrides
        ).filter(pl.col("HADM_ID").is_not_null()).with_columns(pl.col("HADM_ID").cast(pl.Int64)).group_by("HADM_ID").agg([
            pl.col("ITEMID").count().alias("TOTAL_MV_PROCEDURES")
        ])

        # 7. The Master Join
        print("Joining all aggregated tables to the Admissions base...")
        combined = (
            admissions
            .join(diagnoses, on="HADM_ID", how="left")
            .join(prescriptions, on="HADM_ID", how="left")
            .join(lab_events, on="HADM_ID", how="left")
            .join(microbiology, on="HADM_ID", how="left")
            .join(procedures_mv, on="HADM_ID", how="left")
        )

        # 8. Execute the graph to get the actual DataFrame in memory
        print("Executing computation graph...")
        final_df = combined.collect() 
        
        # --- NEW: Flatten Lists to Strings for CSV compatibility ---
        print("Formatting lists for CSV output...")
        final_df = final_df.with_columns([
            pl.col("DIAGNOSIS_NAMES").list.join(" | "),
            pl.col("PRESCRIBED_DRUGS").list.join(" | "),
            pl.col("MICROBIOLOGY_ORGANISMS").list.join(" | ")
        ])

        # 9. Save
        final_df.write_csv(output_path)
        final_df.write_parquet(output_path.replace(".csv", ".parquet"))

        print(f"Success! Final dataset shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
        print(f"Finished in {round(time.time() - start_time, 2)} seconds.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    output_path = "Final Datasets\Complete_MIMIC_Aggregated_Cleaned.csv" # Output path for the final aggregated dataset. Adjust as needed.
    aggregate_all_mimic_data(output_path)