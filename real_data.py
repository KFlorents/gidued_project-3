import pandas as pd

new_csv_filename = "troop_movements_1m.csv"  
df_new = pd.read_csv(new_csv_filename)


df_new["unit_type"] = df_new["unit_type"].replace("invalid_unit", "unknown")
df_new["location_x"] = df_new["location_x"].ffill()
df_new["location_y"] = df_new["location_y"].ffill()

# Save the cleaned dataset
parquet_filename = "troop_movements_1m.parquet"
df_new.to_parquet(parquet_filename, index=False)


