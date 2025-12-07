import pandas as pd
import os

def aggregate_bus_data(file_path, output_dir):
    print(f"Processing file: {file_path}")

    try:
        chunk_size = 500000  
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        
        for i, df in enumerate(chunks):
            print(f"  Processing chunk {i+1}...")

            df['Ride_start_datetime'] = pd.to_datetime(df['Ride_start_date'] + ' ' + df['Ride_start_time'], errors='coerce')

            df.dropna(subset=['Ride_start_datetime'], inplace=True)
            df.set_index('Ride_start_datetime', inplace=True)
            unique_routes = df['Bus_Service_Number'].unique()

            for route in unique_routes:
                route_df = df[df['Bus_Service_Number'] == route]

                start_df = route_df[route_df['Direction'] == 'Start']
                return_df = route_df[route_df['Direction'] == 'Return']

                if not start_df.empty:
                    start_agg = start_df.groupby([
                        pd.Grouper(freq='15T'),
                        'Bus_Service_Number',
                        'Direction',
                        'Boarding_stop_stn',
                        'Alighting_stop_stn'
                    ]).size().reset_index(name='Passenger_Count')

                    output_filename = os.path.join(output_dir, f"{route}_start_aggregated.csv")
                    start_agg.to_csv(output_filename, mode='a', header=not os.path.exists(output_filename), index=False)

                if not return_df.empty:
                    return_agg = return_df.groupby([
                        pd.Grouper(freq='15T'),
                        'Bus_Service_Number',
                        'Direction',
                        'Boarding_stop_stn',
                        'Alighting_stop_stn'
                    ]).size().reset_index(name='Passenger_Count')

                    output_filename = os.path.join(output_dir, f"{route}_return_aggregated.csv")
                    return_agg.to_csv(output_filename, mode='a', header=not os.path.exists(output_filename), index=False)

        print(f"Finished processing {file_path}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    data_directory = './'
    output_directory = 'aggregated_bus_data'

    os.makedirs(output_directory, exist_ok=True)
    
    monthly_files = [
        'BUS_DATA_OCT_2017.csv',
        'BUS_DATA_NOV_2017.csv',
        'BUS_DATA_DEC_2017.csv',
        'BUS_DATA_JAN_2018.csv',
        'BUS_DATA_FEB_2018.csv',
        'BUS_DATA_MAR_2018.csv',
    ]

    for file_name in monthly_files:
        full_file_path = os.path.join(data_directory, file_name)
        aggregate_bus_data(full_file_path, output_directory)

    print(f"\nAll files have been processed. Aggregated data is saved in the '{output_directory}' folder.")
