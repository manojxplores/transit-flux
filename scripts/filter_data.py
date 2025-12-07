import os
import shutil
import pickle

folder = "aggregated_bus_data"
pickle_file = "BusRoutes.pickle"
output_folder = "filtered_bus_data"

os.makedirs(output_folder, exist_ok=True)

with open(pickle_file, "rb") as f:
    ser_dict = pickle.load(f)

allowed_ids = set(ser_dict.keys())

for file in os.listdir(folder):
    if file.endswith(".csv") and file.startswith("SER_"):
        route_id = "_".join(file.split("_")[:2]) 

        if route_id in allowed_ids:
            src = os.path.join(folder, file)
            dst = os.path.join(output_folder, file)
            shutil.copy(src, dst)
            print(f"Copied {file} → {output_folder}")
        else:
            print(f"Skipped {file}")
