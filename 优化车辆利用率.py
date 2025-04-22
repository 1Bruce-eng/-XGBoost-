# -*- coding: utf-8 -*-
# Vehicle Scheduling Script (Refactored)

import pandas as pd
import datetime
import re
import collections
import math # math was missing in the original imports but needed implicitly later if not using integer division
import os
import gc # gc was missing

# === Constants ===
# --- Capacities ---
CAPACITY_NORMAL = 1000  # Capacity without container
CAPACITY_CONTAINER = 800  # Capacity with container

# --- Loading/Unloading Times (hours) ---
TIME_LOAD_UNLOAD_NORMAL = 0.75  # 45 minutes each way
TIME_LOAD_UNLOAD_CONTAINER = 0.167 # Approx 10 minutes each way

# --- Costs ---
FIXED_COST_PER_VEHICLE_DAY = 100 # Not explicitly used in assignment logic below, but defined

# --- Time Windows ---
PAIRING_TIME_WINDOW = datetime.timedelta(minutes=10) # Max time diff for pairing demands

# --- String Constants ---
CONTAINER_YES = 'Y'
CONTAINER_NO = 'N'
EXTERNAL_VEHICLE = '外部' # Keep original term as it appears in output spec

# --- File Paths ---
# Assumes script is in the same directory as '附件' and '结果表' folders
BASE_PATH = '.' # Or specify the correct base path if needed
RESULTS_DIR = os.path.join(BASE_PATH, '结果表')
ATTACHMENTS_DIR = os.path.join(BASE_PATH, '附件')

INPUT_DAILY_SCHEDULE_FILE = os.path.join(RESULTS_DIR, '结果表1.xlsx')
INPUT_10MIN_VOLUMES_FILE = os.path.join(RESULTS_DIR, '结果表2.xlsx')
INPUT_STATION_PAIRS_FILE = os.path.join(ATTACHMENTS_DIR, '附件4.xlsx')
INPUT_FLEET_INFO_FILE = os.path.join(ATTACHMENTS_DIR, '附件5.xlsx')
INPUT_ROUTE_INFO_FILE = os.path.join(ATTACHMENTS_DIR, '附件1.xlsx')
OUTPUT_SCHEDULE_FILE = os.path.join(RESULTS_DIR, '结果表4.xlsx') # Output filename

# === Helper Functions ===

def extract_station_id(station_code):
    """Extracts the numerical ID from a station code string (e.g., '站点83' -> 83)."""
    if not isinstance(station_code, str):
        return None
    match = re.search(r'站点(\d+)', station_code)
    return int(match.group(1)) if match else None

def load_pairing_rules(filepath):
    """Loads valid station pairing rules from the Excel file."""
    try:
        df = pd.read_excel(filepath)
        # Extract IDs robustly, handling potential non-string entries
        pairs = set()
        for _, row in df.iterrows():
            id1 = extract_station_id(str(row.iloc[0])) # Assuming first two columns
            id2 = extract_station_id(str(row.iloc[1]))
            if id1 is not None and id2 is not None:
                pairs.add(tuple(sorted((id1, id2)))) # Store sorted tuple to handle (A,B) and (B,A)
        # Add reverse pairs for easier lookup if needed (original code did this)
        # pairs |= {(b, a) for a, b in pairs}
        # Storing sorted tuples makes the reverse redundant for checking.
        print(f"Loaded {len(pairs)} unique station pairing rules.")
        return pairs
    except FileNotFoundError:
        print(f"Error: Pairing rules file not found at {filepath}")
        return set()
    except Exception as e:
        print(f"Error loading pairing rules from {filepath}: {e}")
        return set()

def load_route_dates(filepath):
    """Loads the reference date for each route from the daily schedule file."""
    try:
        df = pd.read_excel(filepath)
        route_dates = {
            row['线路编码']: pd.to_datetime(row['日期']).date()
            for _, row in df.iterrows()
            if '线路编码' in row and '日期' in row # Basic check
        }
        print(f"Loaded reference dates for {len(route_dates)} routes.")
        return route_dates
    except FileNotFoundError:
        print(f"Error: Daily schedule file not found at {filepath}")
        return {}
    except Exception as e:
        print(f"Error loading route dates from {filepath}: {e}")
        return {}

def calculate_vehicle_return_time(demand_info):
    """Calculates the return time for a vehicle completing a demand."""
    load_unload_time = TIME_LOAD_UNLOAD_CONTAINER if demand_info['use_container'] else TIME_LOAD_UNLOAD_NORMAL
    # Total time = depart_time + load_time + travel_to + unload_time + travel_back
    # Assuming symmetrical travel and load/unload at both ends
    total_duration_hours = (2 * load_unload_time) + (2 * demand_info['route_duration_hours'])
    return demand_info['depart_datetime'] + datetime.timedelta(hours=total_duration_hours)


# === Main Logic ===

# --- 1. Load Input Data ---
print("Loading input data...")
try:
    df_10min_volumes = pd.read_excel(INPUT_10MIN_VOLUMES_FILE)
    df_route_info = pd.read_excel(INPUT_ROUTE_INFO_FILE)
    df_fleet_info = pd.read_excel(INPUT_FLEET_INFO_FILE)
except FileNotFoundError as e:
    print(f"Error: Input file not found - {e}. Exiting.")
    exit()
except Exception as e:
    print(f"Error reading input files: {e}. Exiting.")
    exit()

# Load supporting data using helper functions
valid_station_pairs = load_pairing_rules(INPUT_STATION_PAIRS_FILE)
route_departure_dates = load_route_dates(INPUT_DAILY_SCHEDULE_FILE)
fleet_capacities = df_fleet_info.set_index('车队编码')['自有车数量'].to_dict()

# --- 2. Generate Initial Demands (800 unit granularity) ---
print("Generating initial demands based on container capacity...")
initial_demands = []
df_10min_volumes['datetime'] = pd.to_datetime(df_10min_volumes['日期'].astype(str) + ' ' + df_10min_volumes['分钟起始'].astype(str))

for route_code, group in df_10min_volumes.groupby('线路编码'):
    group = group.sort_values('datetime')
    try:
        route_info = df_route_info[df_route_info['线路编码'] == route_code].iloc[0]
        # Safely get departure node time, handling potential format issues
        departure_node_time_str = route_info['发运节点'] if isinstance(route_info['发运节点'], str) else route_info['发运节点'].strftime('%H:%M:%S')
        departure_time_obj = datetime.time.fromisoformat(departure_node_time_str)
        departure_date = route_departure_dates.get(route_code)
        if departure_date is None:
            print(f"Warning: No reference date found for route {route_code}. Skipping.")
            continue
        base_departure_dt = datetime.datetime.combine(departure_date, departure_time_obj)

    except (IndexError, KeyError, ValueError, TypeError) as e:
        print(f"Warning: Could not process route info for {route_code}. Error: {e}. Skipping.")
        continue

    accumulated_volume = 0
    for _, row in group.iterrows():
        volume = int(row['包裹量'] or 0)
        accumulated_volume += volume
        current_dt = row['datetime']

        # Create demand when container capacity is reached
        while accumulated_volume >= CAPACITY_CONTAINER:
            initial_demands.append({
                'route_codes': [route_code], # Store as list for potential merging
                'volume': CAPACITY_CONTAINER,
                'depart_datetime': current_dt, # Depart when full
                'fleet_id': route_info['车队编码'],
                'route_duration_hours': float(route_info['在途时长']),
                'variable_cost': float(route_info['自有变动成本']), # Renamed for clarity
                'external_cost': float(route_info['外部承运商成本']),# Renamed for clarity
                'use_container': True, # Default to using container
                'origin_station': route_info['起始场地'],
                'destination_station': route_info['目的场地']
            })
            accumulated_volume -= CAPACITY_CONTAINER

    # Add remaining volume as a separate demand, departing at scheduled node time
    if accumulated_volume > 0:
        initial_demands.append({
            'route_codes': [route_code],
            'volume': accumulated_volume,
            'depart_datetime': base_departure_dt, # Leftovers depart at scheduled time
            'fleet_id': route_info['车队编码'],
            'route_duration_hours': float(route_info['在途时长']),
            'variable_cost': float(route_info['自有变动成本']),
            'external_cost': float(route_info['外部承运商成本']),
            'use_container': True, # Still default to container
            'origin_station': route_info['起始场地'],
            'destination_station': route_info['目的场地']
        })
print(f"Generated {len(initial_demands)} initial demands.")

# --- 3. Pair Demands (串点 - Combine routes within 10 min, same fleet, pairable stations, <= 800 vol) ---
print("Attempting to pair demands (串点)...")
demands_after_pairing = []
processed_indices = [False] * len(initial_demands)
initial_demands_sorted = sorted(initial_demands, key=lambda x: (x['fleet_id'], x['depart_datetime'])) # Sort for efficient pairing

for i, demand1 in enumerate(initial_demands_sorted):
    if processed_indices[i] or demand1['volume'] == CAPACITY_CONTAINER: # Skip if processed or already full
        if not processed_indices[i]: # Ensure full containers are added if not part of a pair
             demands_after_pairing.append(demand1)
             processed_indices[i] = True
        continue

    paired = False
    # Look ahead for potential pairs within the time window
    for j in range(i + 1, len(initial_demands_sorted)):
        demand2 = initial_demands_sorted[j]

        # Optimization: If we've passed the time window or fleet changes, stop searching for this demand1
        if demand2['fleet_id'] != demand1['fleet_id'] or \
           (demand2['depart_datetime'] - demand1['depart_datetime']) > PAIRING_TIME_WINDOW:
            break

        if processed_indices[j]: continue # Skip already processed

        # --- Check Pairing Conditions ---
        # 1. Volume Constraint
        if demand1['volume'] + demand2['volume'] > CAPACITY_CONTAINER:
            continue

        # 2. Station Pairing Constraint (Check all combinations)
        stations1_ids = {extract_station_id(s) for s in [demand1['origin_station'], demand1['destination_station']]}
        stations2_ids = {extract_station_id(s) for s in [demand2['origin_station'], demand2['destination_station']]}
        all_involved_station_ids = stations1_ids | stations2_ids
        # Check if *any* pair formed by combining these is invalid.
        # A simpler check might be needed depending on exact pairing logic (e.g., is it based on combining destinations?)
        # Original logic checked all pairs between lines, let's adapt to stations
        # Assuming pairing means the origin/dest of one can be paired with origin/dest of another
        # This check might need refinement based on the exact business rule for "串点" feasibility
        is_pairable = True
        involved_ids_list = [id for id in all_involved_station_ids if id is not None]
        if len(involved_ids_list) > 1: # Need at least two stations to check pairing
             for s1_idx in range(len(involved_ids_list)):
                  for s2_idx in range(s1_idx + 1, len(involved_ids_list)):
                       pair_to_check = tuple(sorted((involved_ids_list[s1_idx], involved_ids_list[s2_idx])))
                       if pair_to_check not in valid_station_pairs:
                            is_pairable = False; break
                  if not is_pairable: break

        if not is_pairable:
             continue

        # --- Merge Demands ---
        merged_demand = demand1.copy() # Start with demand1's info
        merged_demand.update({
            'route_codes': list(set(demand1['route_codes'] + demand2['route_codes'])), # Combine unique routes
            'volume': demand1['volume'] + demand2['volume'],
            'depart_datetime': max(demand1['depart_datetime'], demand2['depart_datetime']), # Depart later
            'route_duration_hours': max(demand1['route_duration_hours'], demand2['route_duration_hours']), # Takes longer route time
            'variable_cost': max(demand1['variable_cost'], demand2['variable_cost']), # Assume higher cost applies
            'external_cost': max(demand1['external_cost'], demand2['external_cost']), # Assume higher cost applies
            # Keep origin/destination simple for now - might need complex logic
            'origin_station': demand1['origin_station'], # Placeholder
            'destination_station': demand1['destination_station'] # Placeholder
        })
        demands_after_pairing.append(merged_demand)
        processed_indices[i] = processed_indices[j] = True
        paired = True
        break # Stop searching for pairs for demand1

    if not paired: # If demand1 couldn't be paired
        demands_after_pairing.append(demand1)
        processed_indices[i] = True

print(f"Demands after pairing: {len(demands_after_pairing)}.")

# --- 4. Decontainerize & Merge (Combine container trips to non-container if vol <= 1000) ---
print("Attempting to remove containers and merge trips...")
# Group by fleet and 10-minute departure bucket for efficient searching
demands_by_key = collections.defaultdict(list)
for idx, d in enumerate(demands_after_pairing):
    # Create a time bucket key (floor to 10 minutes)
    bucket_time = d['depart_datetime'].replace(minute=(d['depart_datetime'].minute // 10) * 10, second=0, microsecond=0)
    key = (d['fleet_id'], bucket_time)
    demands_by_key[key].append(idx)

final_demands = []
processed_indices_stage3 = [False] * len(demands_after_pairing)

for idx1, demand1 in enumerate(demands_after_pairing):
    if processed_indices_stage3[idx1]: continue

    merged_in_stage3 = False
    # Try merging only if it's currently using a container and has room to grow
    if demand1['use_container'] and demand1['volume'] < CAPACITY_CONTAINER:
        bucket_time = demand1['depart_datetime'].replace(minute=(demand1['depart_datetime'].minute // 10) * 10, second=0, microsecond=0)
        key = (demand1['fleet_id'], bucket_time)
        candidate_indices = demands_by_key.get(key, [])

        for idx2 in candidate_indices:
            if idx1 == idx2 or processed_indices_stage3[idx2]: continue

            demand2 = demands_after_pairing[idx2]

            # --- Check Decontainerization Conditions ---
            # 1. Both must currently use containers
            if not demand2['use_container']: continue
            # 2. Combined volume must be <= NORMAL capacity
            if demand1['volume'] + demand2['volume'] > CAPACITY_NORMAL: continue
            # 3. Assume fleets already match due to key structure

            # --- Merge and Remove Container ---
            merged_demand = demand1.copy()
            merged_demand.update({
                'route_codes': list(set(demand1['route_codes'] + demand2['route_codes'])),
                'volume': demand1['volume'] + demand2['volume'],
                'depart_datetime': max(demand1['depart_datetime'], demand2['depart_datetime']),
                'route_duration_hours': max(demand1['route_duration_hours'], demand2['route_duration_hours']),
                'variable_cost': max(demand1['variable_cost'], demand2['variable_cost']),
                'external_cost': max(demand1['external_cost'], demand2['external_cost']),
                'use_container': False, # Key change: No longer using container
                'origin_station': demand1['origin_station'], # Placeholder
                'destination_station': demand1['destination_station'] # Placeholder
            })
            final_demands.append(merged_demand)
            processed_indices_stage3[idx1] = processed_indices_stage3[idx2] = True
            merged_in_stage3 = True
            break # Stop searching for partners for demand1

    if not merged_in_stage3: # If not merged in this stage, add the demand as is
        final_demands.append(demand1)
        processed_indices_stage3[idx1] = True

print(f"Final demands after decontainerization attempts: {len(final_demands)}.")

# --- 5. Vehicle Scheduling & Dynamic Container Adjustment ---
print("Assigning vehicles to demands...")
# Initialize vehicle availability tracking
# Structure: {fleet_id: [{'id': vehicle_id, 'return_time': datetime}, ...]}
fleet_vehicle_status = {fleet_id: [] for fleet_id in fleet_capacities}
vehicle_sequence_num = {fleet_id: 1 for fleet_id in fleet_capacities} # For naming new vehicles
assignments = []

# Process demands sorted by departure time
for demand in sorted(final_demands, key=lambda x: x['depart_datetime']):
    fleet_id = demand['fleet_id']
    departure_dt = demand['depart_datetime']
    assigned_vehicle_id = None
    is_external = False

    # --- 1. Try to find an available existing vehicle in the fleet ---
    best_available_vehicle = None
    for vehicle in fleet_vehicle_status.get(fleet_id, []):
        if vehicle['return_time'] <= departure_dt:
             # Found an available vehicle
             if best_available_vehicle is None or vehicle['return_time'] < best_available_vehicle['return_time']:
                  # Choose the one returning earliest if multiple are free
                  best_available_vehicle = vehicle

    if best_available_vehicle:
        assigned_vehicle_id = best_available_vehicle['id']
        best_available_vehicle['return_time'] = calculate_vehicle_return_time(demand) # Update return time
        print(f"  Assigned existing vehicle {assigned_vehicle_id} to demand departing at {departure_dt.strftime('%Y-%m-%d %H:%M')}")

    # --- 2. If no existing vehicle, try allocating a new one if fleet capacity allows ---
    elif fleet_id in fleet_capacities and len(fleet_vehicle_status[fleet_id]) < fleet_capacities[fleet_id]:
        new_vehicle_id = f"{fleet_id}-V{vehicle_sequence_num[fleet_id]:02d}"
        vehicle_sequence_num[fleet_id] += 1
        new_vehicle = {'id': new_vehicle_id, 'return_time': calculate_vehicle_return_time(demand)}
        fleet_vehicle_status[fleet_id].append(new_vehicle)
        assigned_vehicle_id = new_vehicle_id
        print(f"  Allocated new vehicle {assigned_vehicle_id} to demand departing at {departure_dt.strftime('%Y-%m-%d %H:%M')}")

    # --- 3. Dynamic Container Adjustment Attempt ---
    # If still no vehicle and the demand *could* use a container (vol <= 800) but isn't,
    # try switching to container to see if the shorter time frees up a vehicle.
    elif not demand['use_container'] and demand['volume'] <= CAPACITY_CONTAINER:
        demand['use_container'] = True # Tentatively switch
        print(f"  Attempting dynamic container switch for demand departing at {departure_dt.strftime('%Y-%m-%d %H:%M')}")
        # Re-check for available vehicles with the potentially earlier return time
        best_available_after_switch = None
        for vehicle in fleet_vehicle_status.get(fleet_id, []):
            if vehicle['return_time'] <= departure_dt: # Check availability at departure time
                if best_available_after_switch is None or vehicle['return_time'] < best_available_after_switch['return_time']:
                    best_available_after_switch = vehicle

        if best_available_after_switch:
            assigned_vehicle_id = best_available_after_switch['id']
            best_available_after_switch['return_time'] = calculate_vehicle_return_time(demand) # Use updated return time
            print(f"    Success! Assigned existing vehicle {assigned_vehicle_id} after switching to container.")
        else:
            # Switch failed to free up a vehicle, revert the change (optional, but cleaner)
            demand['use_container'] = False
            print(f"    Switching to container did not free up a vehicle.")
            # Proceed to assign external vehicle

    # --- 4. Assign External Vehicle ---
    if assigned_vehicle_id is None:
        assigned_vehicle_id = EXTERNAL_VEHICLE
        is_external = True
        print(f"  Assigned EXTERNAL vehicle to demand departing at {departure_dt.strftime('%Y-%m-%d %H:%M')}")


    # Record the assignment details
    assignments.append({
        'demand_info': demand,
        'assigned_vehicle': assigned_vehicle_id,
        'is_external': is_external
    })
    # Sort vehicles by return time within each fleet for efficient searching next time
    if fleet_id in fleet_vehicle_status:
         fleet_vehicle_status[fleet_id].sort(key=lambda v: v['return_time'])


# --- 6. Generate Output Table (结果表4) ---
print("Generating output schedule (结果表4)...")
output_rows = []
for assignment in assignments:
    demand_info = assignment['demand_info']
    output_rows.append({
        '线路编码': ' + '.join(demand_info['route_codes']), # Join multiple route codes if merged
        '日期': demand_info['depart_datetime'].date().strftime('%Y-%m-%d'),
        '预计发运时间': demand_info['depart_datetime'].strftime('%H:%M'),
        '是否使用容器': CONTAINER_YES if demand_info['use_container'] else CONTAINER_NO,
        '发运车辆': assignment['assigned_vehicle']
    })

output_df = pd.DataFrame(output_rows)

# --- 7. Save Output ---
try:
    output_df.to_excel(OUTPUT_SCHEDULE_FILE, index=False)
    print(f"Successfully saved schedule to {OUTPUT_SCHEDULE_FILE}")
except Exception as e:
    print(f"Error: Failed to save output file {OUTPUT_SCHEDULE_FILE}: {e}")

print("\n--- Script execution finished ---")