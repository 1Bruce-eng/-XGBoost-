import pandas as pd
import datetime
import re
import itertools
import os
from typing import List, Dict, Tuple, Set, Optional, Any

# === Constants ===
TRUCK_CAPACITY = 1000           # Single truck loading capacity (units: packages)
LOAD_UNLOAD_TIME_EACH = 0.75    # Loading/Unloading time each (units: hours)
LOAD_UNLOAD_HOURS_TOTAL = LOAD_UNLOAD_TIME_EACH * 2 # Total loading and unloading time per trip
MERGE_TIME_WINDOW_MINUTES = 10  # Max time difference for considering demand merging

# --- File Paths ---
RESULTS_DIR = "./结果表/"
ATTACHMENTS_DIR = "./附件/"
DAILY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "结果表1.xlsx")
TEN_MIN_INTERVAL_FILE = os.path.join(RESULTS_DIR, "结果表2.xlsx")
PAIRABLE_STATIONS_FILE = os.path.join(ATTACHMENTS_DIR, "附件4.xlsx")
FLEET_INFO_FILE = os.path.join(ATTACHMENTS_DIR, "附件5.xlsx")
LINE_INFO_FILE = os.path.join(ATTACHMENTS_DIR, "附件1.xlsx")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "结果表3.xlsx")

# --- Column Names (using constants for robustness) ---
COL_LINE_CODE = '线路编码'
COL_DATE = '日期'
COL_START_MINUTE = '分钟起始'
COL_PACKAGE_VOLUME = '包裹量'
COL_STATION_CODE_1 = '站点编号1'
COL_STATION_CODE_2 = '站点编号2'
COL_DEPARTURE_NODE = '发运节点'
COL_FLEET_CODE = '车队编码'
COL_TRANSIT_HOURS = '在途时长'
COL_INTERNAL_VAR_COST = '自有变动成本'
COL_EXTERNAL_COST = '外部承运商成本'
COL_FLEET_SIZE = '自有车数量'
COL_OUTPUT_DEPART_TIME = '预计发运时间'
COL_OUTPUT_VEHICLE = '发运车辆'


# === Utility Functions ===
def get_station_id(station_code: str) -> Optional[int]:
    """Extracts the numerical station ID from a string like '站点123'."""
    match = re.search(r'站点(\d+)', str(station_code)) # Ensure input is string
    return int(match.group(1)) if match else None

def calculate_turnaround_time(route_hours: float) -> datetime.timedelta:
    """Calculates the total turnaround time for a vehicle."""
    return datetime.timedelta(hours=(LOAD_UNLOAD_HOURS_TOTAL + 2 * route_hours))

# === Data Loading ===
print("Loading data...")
try:
    df_daily_summary = pd.read_excel(DAILY_SUMMARY_FILE)
    df_10min_volume = pd.read_excel(TEN_MIN_INTERVAL_FILE)
    df_pairable_stations = pd.read_excel(PAIRABLE_STATIONS_FILE)
    df_fleet_info = pd.read_excel(FLEET_INFO_FILE)
    df_line_info = pd.read_excel(LINE_INFO_FILE)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure input files are in the correct directories.")
    exit(1)

# === Step 1: Prepare Pairable Station Set ===
# Create a set of allowed station pairs for efficient lookup.
# A pair (s1, s2) is allowed if (s1, s2) or (s2, s1) is in the input file.
pairable_station_set: Set[Tuple[int, int]] = set()
for _, row in df_pairable_stations.iterrows():
    s1 = get_station_id(row[COL_STATION_CODE_1])
    s2 = get_station_id(row[COL_STATION_CODE_2])
    if s1 is not None and s2 is not None:
        pairable_station_set.add((s1, s2))
        pairable_station_set.add((s2, s1)) # Add reverse pair for easy checking
print(f"Loaded {len(pairable_station_set)//2} unique pairable station combinations.")

# === Step 2: Create Line-to-Date Mapping ===
# Use the daily summary to assign a primary operational date to each line.
# This helps normalize dates for multi-day operations read from the 10min data.
line_to_primary_date_map: Dict[str, datetime.date] = {
    row[COL_LINE_CODE]: pd.to_datetime(row[COL_DATE]).date()
    for _, row in df_daily_summary.iterrows()
}

# === Step 3: Process 10-Minute Interval Data ===
# Convert date and start minute into a proper datetime object.
df_10min_volume['datetime'] = df_10min_volume.apply(
    lambda r: pd.to_datetime(str(r[COL_DATE])) + pd.to_timedelta(str(r[COL_START_MINUTE])),
    axis=1
)
# Ensure package volume is numeric, handling potential NaNs
df_10min_volume[COL_PACKAGE_VOLUME] = pd.to_numeric(df_10min_volume[COL_PACKAGE_VOLUME], errors='coerce').fillna(0).astype(int)


# === Step 4: Generate Initial Transportation Demands ===
# Create individual transport demands based on accumulated volume reaching truck capacity
# or the final scheduled departure time.
initial_demands: List[Dict[str, Any]] = []
print("Generating initial demands...")
for line_code, group in df_10min_volume.groupby(COL_LINE_CODE):
    group = group.sort_values('datetime')
    try:
        line_info = df_line_info[df_line_info[COL_LINE_CODE] == line_code].iloc[0]
    except IndexError:
        print(f"Warning: No line info found for {line_code} in {LINE_INFO_FILE}. Skipping this line.")
        continue

    # Determine the scheduled departure datetime for the last truck of the day
    departure_node_time_str = line_info[COL_DEPARTURE_NODE]
    departure_time = (datetime.time.fromisoformat(departure_node_time_str)
                      if isinstance(departure_node_time_str, str)
                      else departure_node_time_str) # Handle if it's already time object

    # Use the date from the daily summary if available, otherwise fallback to the last date in the 10min data
    primary_date = line_to_primary_date_map.get(line_code, group['datetime'].iloc[-1].date())
    scheduled_departure_dt = datetime.datetime.combine(primary_date, departure_time)

    accumulated_volume = 0
    for _, row in group.iterrows():
        accumulated_volume += row[COL_PACKAGE_VOLUME]

        # If accumulated volume fills a truck before the scheduled time, dispatch it
        while accumulated_volume >= TRUCK_CAPACITY:
            initial_demands.append({
                'lines': [line_code],                 # List of lines in this demand
                'volume': TRUCK_CAPACITY,             # Volume for this truck
                'depart_dt': row['datetime'],         # Departure time (when capacity was reached)
                'fleet': line_info[COL_FLEET_CODE],   # Assigned fleet
                'route_hours': line_info[COL_TRANSIT_HOURS], # Transit duration
                'internal_v_cost': line_info[COL_INTERNAL_VAR_COST], # Cost if internal fleet used
                'external_cost': line_info[COL_EXTERNAL_COST], # Cost if external fleet used
            })
            accumulated_volume -= TRUCK_CAPACITY

    # Add the remaining volume as a final demand departing at the scheduled time
    if accumulated_volume > 0:
        initial_demands.append({
            'lines': [line_code],
            'volume': accumulated_volume,
            'depart_dt': scheduled_departure_dt, # Final departure at scheduled time
            'fleet': line_info[COL_FLEET_CODE],
            'route_hours': line_info[COL_TRANSIT_HOURS],
            'internal_v_cost': line_info[COL_INTERNAL_VAR_COST],
            'external_cost': line_info[COL_EXTERNAL_COST],
        })
print(f"Generated {len(initial_demands)} initial demands.")

# === Step 5: Merge Demands (Greedy Pairing Heuristic) ===
# Attempt to merge pairs of demands if they meet criteria:
# - Depart close in time (within MERGE_TIME_WINDOW_MINUTES)
# - Belong to the same fleet
# - All station pairs between the two demands are pairable
# - Combined volume does not exceed TRUCK_CAPACITY
# - Neither demand is already a full truckload

print("Attempting to merge demands...")
initial_demands.sort(key=lambda d: d['depart_dt']) # Sort by departure time for efficient windowing
merged_demands: List[Dict[str, Any]] = []
demand_processed: List[bool] = [False] * len(initial_demands)
merge_time_window = datetime.timedelta(minutes=MERGE_TIME_WINDOW_MINUTES)

for i, current_demand in enumerate(initial_demands):
    if demand_processed[i] or current_demand['volume'] >= TRUCK_CAPACITY:
        if not demand_processed[i]: # Add unprocessed full trucks
             merged_demands.append(current_demand)
             demand_processed[i] = True
        continue

    merged_successfully = False
    for j in range(i + 1, len(initial_demands)):
        candidate_demand = initial_demands[j]

        # Early exit if candidate is too far in the future
        if (candidate_demand['depart_dt'] - current_demand['depart_dt']) > merge_time_window:
            break

        if demand_processed[j] or candidate_demand['volume'] >= TRUCK_CAPACITY: continue
        if current_demand['fleet'] != candidate_demand['fleet']: continue

        # Check station pairability
        stations1 = {get_station_id(line) for line in current_demand['lines']}
        stations2 = {get_station_id(line) for line in candidate_demand['lines']}
        stations1.discard(None) # Remove None if get_station_id failed
        stations2.discard(None)

        is_pairable = True
        if not stations1 or not stations2: # Cannot merge if station IDs are missing
             is_pairable = False
        else:
            for s1, s2 in itertools.product(stations1, stations2):
                if (s1, s2) not in pairable_station_set:
                    is_pairable = False
                    break
        if not is_pairable: continue

        # Check combined volume
        if current_demand['volume'] + candidate_demand['volume'] > TRUCK_CAPACITY: continue

        # --- Merge is possible ---
        merged_demand_data = {
            'lines': current_demand['lines'] + candidate_demand['lines'],
            'volume': current_demand['volume'] + candidate_demand['volume'],
            'depart_dt': max(current_demand['depart_dt'], candidate_demand['depart_dt']), # Depart at the later time
            'fleet': current_demand['fleet'],
            # Use max for route/cost as a conservative heuristic (could be averaged or summed depending on context)
            'route_hours': max(current_demand['route_hours'], candidate_demand['route_hours']),
            'internal_v_cost': max(current_demand['internal_v_cost'], candidate_demand['internal_v_cost']),
            'external_cost': max(current_demand['external_cost'], candidate_demand['external_cost']),
        }
        merged_demands.append(merged_demand_data)
        demand_processed[i] = True
        demand_processed[j] = True
        merged_successfully = True
        break # Stop searching for merges for current_demand (greedy approach)

    # If no merge occurred for current_demand, add it as is
    if not merged_successfully:
        merged_demands.append(current_demand)
        demand_processed[i] = True

print(f"Reduced to {len(merged_demands)} demands after merging.")

# === Step 6: Assign Vehicles ===
# Assign vehicles from the internal fleet if available and capacity allows,
# otherwise use external carriers. Prioritize reusing vehicles.

print("Assigning vehicles to demands...")
# Fleet capacity dictionary: {fleet_code: number_of_trucks}
fleet_capacities = df_fleet_info.set_index(COL_FLEET_CODE)[COL_FLEET_SIZE].to_dict()

# Track available internal vehicles for each fleet and their return time
# Structure: {fleet_code: [{'id': vehicle_id, 'return_dt': datetime}, ...]}
fleet_vehicle_pool: Dict[str, List[Dict[str, Any]]] = {fleet_code: [] for fleet_code in fleet_capacities}
# Track next vehicle ID number for each fleet
vehicle_sequence_num: Dict[str, int] = {fleet_code: 1 for fleet_code in fleet_capacities}

assignments: List[Tuple[Dict[str, Any], str, bool]] = [] # Stores (demand_dict, vehicle_id, is_external)

# Sort merged demands by departure time for chronological assignment
merged_demands.sort(key=lambda d: d['depart_dt'])

for demand in merged_demands:
    fleet_code = demand['fleet']
    departure_dt = demand['depart_dt']
    assigned_vehicle_id = None
    is_external = False

    # --- Try to find a reusable internal vehicle ---
    best_reusable_vehicle_info = None
    earliest_return_among_available = datetime.datetime.max # Find vehicle that becomes available earliest but before required departure

    if fleet_code in fleet_vehicle_pool:
        available_vehicles = [
            v for v in fleet_vehicle_pool[fleet_code] if v['return_dt'] <= departure_dt
        ]
        if available_vehicles:
            # Simple strategy: use the one that returned earliest (FIFO-like reuse)
            best_reusable_vehicle_info = min(available_vehicles, key=lambda v: v['return_dt'])

        if best_reusable_vehicle_info:
            assigned_vehicle_id = best_reusable_vehicle_info['id']
            # Update its return time for the new trip
            turnaround_duration = calculate_turnaround_time(demand['route_hours'])
            best_reusable_vehicle_info['return_dt'] = departure_dt + turnaround_duration

    # --- If no reusable vehicle, try to use a new internal vehicle if capacity allows ---
    if assigned_vehicle_id is None and fleet_code in fleet_capacities:
        current_fleet_size = len(fleet_vehicle_pool.get(fleet_code, []))
        max_fleet_size = fleet_capacities.get(fleet_code, 0)

        if current_fleet_size < max_fleet_size:
            # Create and assign a new internal vehicle
            new_vehicle_num = vehicle_sequence_num[fleet_code]
            assigned_vehicle_id = f"{fleet_code}-V{new_vehicle_num:02d}"
            vehicle_sequence_num[fleet_code] += 1

            turnaround_duration = calculate_turnaround_time(demand['route_hours'])
            new_vehicle_info = {
                'id': assigned_vehicle_id,
                'return_dt': departure_dt + turnaround_duration
            }
            if fleet_code not in fleet_vehicle_pool:
                 fleet_vehicle_pool[fleet_code] = []
            fleet_vehicle_pool[fleet_code].append(new_vehicle_info)


    # --- If no internal vehicle available, assign an external one ---
    if assigned_vehicle_id is None:
        assigned_vehicle_id = '外部' # External carrier identifier
        is_external = True

    assignments.append((demand, assigned_vehicle_id, is_external))

print(f"Assigned vehicles to {len(assignments)} demands.")

# === Step 7: Generate Output File (结果表3) ===
print(f"Generating output file: {OUTPUT_FILE}...")
output_rows = []
for demand_info, vehicle_id, _ in assignments:
    # Use the primary date associated with the *first* line in the demand, if available
    primary_line = demand_info['lines'][0]
    tag_date = line_to_primary_date_map.get(primary_line, demand_info['depart_dt'].date())

    output_rows.append({
        COL_LINE_CODE: ' + '.join(demand_info['lines']),
        COL_DATE: tag_date.strftime('%Y-%m-%d'),
        COL_OUTPUT_DEPART_TIME: demand_info['depart_dt'].strftime('%H:%M'),
        COL_OUTPUT_VEHICLE: vehicle_id
    })

df_output = pd.DataFrame(output_rows)

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save the output Excel file
try:
    df_output.to_excel(OUTPUT_FILE, index=False, engine='openpyxl') # Specify engine if needed
    print(f"Output file saved successfully to {OUTPUT_FILE}.")
except Exception as e:
    print(f"Error saving output file: {e}")