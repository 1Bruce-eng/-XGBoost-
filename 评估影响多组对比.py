# -*- coding: utf-8 -*-
# Vehicle Dispatch Simulation Script (Refactored - Volume Scenarios)

import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Needed for plotting
import re
import collections
import math
import gc

# --- Configuration: Constants ---
CAPACITY_NORMAL = 1000          # Vehicle capacity without container (units)
CAPACITY_CONTAINER = 800        # Vehicle capacity with container (units)
# Assuming 0.75 and 10/60 are ONE-WAY load/unload times
LOAD_TIME_NORMAL = 0.75         # One-way load/unload time without container (hours, 45 min)
LOAD_TIME_CONTAINER = 10 / 60   # One-way load/unload time with container (hours, 10 min)
PAIRING_WINDOW_MINUTES = 10     # Max time difference for pairing demands (minutes)
FIXED_COST_PER_OWNED_VEHICLE = 100 # Daily fixed cost per owned vehicle (currency)
ASSUMED_EXTERNAL_COST_PER_TRIP = 200 # Assumed cost for an external vehicle trip (currency)
EXTERNAL_VEHICLE_ID = '外部'       # Identifier for external vehicles

# --- Configuration: File Paths ---
# Assumes script is in the same directory as '附件' and '结果表' folders
BASE_PATH = '.'
RESULTS_DIR = os.path.join(BASE_PATH, '结果表')
ATTACHMENTS_DIR = os.path.join(BASE_PATH, '附件')

# Input Files
DAILY_SCHEDULE_FILE = os.path.join(RESULTS_DIR, '结果表1.xlsx')      # Provides dates per route
TEN_MIN_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, '结果表2.xlsx') # Predicted volumes
STATION_PAIRS_FILE = os.path.join(ATTACHMENTS_DIR, '附件4.xlsx')     # Allowed station pairs for multi-stop
FLEET_INFO_FILE = os.path.join(ATTACHMENTS_DIR, '附件5.xlsx')        # Own fleet sizes
ROUTE_INFO_FILE = os.path.join(ATTACHMENTS_DIR, '附件1.xlsx')        # Route details (times, costs, etc.)

# Output Files (simulation results summary and plot)
OUTPUT_SUMMARY_FILE = os.path.join(RESULTS_DIR, '仿真结果汇总_优化版.xlsx') # New output file for summary table
OUTPUT_PLOT_FILE = os.path.join(RESULTS_DIR, '仿真结果对比图_优化版.png') # Output plot file

# --- Load Base Data (Global Scope - as in original) ---
print("Loading base data...")
try:
    df_daily_schedule = pd.read_excel(DAILY_SCHEDULE_FILE)
    df_ten_min_predictions = pd.read_excel(TEN_MIN_PREDICTIONS_FILE)
    df_station_pairs = pd.read_excel(STATION_PAIRS_FILE)
    df_fleet_info = pd.read_excel(FLEET_INFO_FILE)
    df_route_info = pd.read_excel(ROUTE_INFO_FILE) # df_info in original code
    print("Base data loaded successfully.")
except FileNotFoundError as e:
    print(f"Fatal Error: Input file not found - {e}. Exiting.")
    exit()
except Exception as e:
    print(f"Fatal Error reading input files: {e}. Exiting.")
    exit()

# --- Helper Functions (Data Processing) ---

def extract_station_id(code_str):
    """Extracts the numerical ID from a station code string (e.g., '站点83' -> 83)."""
    if not isinstance(code_str, str): return None
    match = re.search(r"站点(\d+)", code_str)
    return int(match.group(1)) if match else None

# Pre-process global data
fleet_capacities = df_fleet_info.set_index("车队编码")["自有车数量"].to_dict()
# Create mapping from route code to its base date
route_base_dates = {
    row["线路编码"]: pd.to_datetime(row["日期"]).date()
    for _, row in df_daily_schedule.iterrows()
}
# Create set of allowed station pair IDs
allowed_station_pairs_ids = set()
for _, row in df_station_pairs.iterrows():
    s1_id = extract_station_id(str(row.iloc[0]))
    s2_id = extract_station_id(str(row.iloc[1]))
    if s1_id is not None and s2_id is not None:
        allowed_station_pairs_ids.add((s1_id, s2_id))
        allowed_station_pairs_ids.add((s2_id, s1_id)) # Add reverse for symmetric check

def extract_station_id(code_str):
    """Extracts the numerical ID from a station code string (e.g., '站点83' -> 83)."""
    if not isinstance(code_str, str): return None
    match = re.search(r"站点(\d+)", code_str)
    return int(match.group(1)) if match else None

def parse_time_robustly(time_value):
    """Converts various time representations to a datetime.time object."""
    if isinstance(time_value, datetime.time): return time_value
    if isinstance(time_value, str):
        try: return datetime.time.fromisoformat(time_value.split('.')[0]) # Handle potential microseconds
        except ValueError: pass
    try: return pd.to_datetime(time_value).time()
    except Exception: return None

def calculate_turnaround_duration(demand_info):
    """Calculates the total round-trip duration for a vehicle completing a demand."""
    # Load/unload time is for ONE-WAY, so multiply by 2 for round trip
    load_unload_time_total = 2 * (LOAD_TIME_CONTAINER if demand_info["use_container"] else LOAD_TIME_NORMAL)
    # Travel time is also one-way route duration, multiply by 2 for round trip
    travel_time_total = 2 * demand_info["route_duration_hours"]
    # Ensure duration is non-negative, though calculation should prevent negative values
    return datetime.timedelta(hours=(travel_time_total + load_unload_time_total))


# --- Simulation Stages (Core Logic Functions) ---

def generate_initial_demands(df_predictions, df_route_details, route_base_dates, volume_multiplier):
    """Generates initial demands based on 10-minute predictions and container capacity."""
    df = df_predictions.copy()
    df["包裹量"] = (df["包裹量"] * volume_multiplier).round().astype(int)
    df["datetime"] = pd.to_datetime(df["日期"].astype(str) + ' ' + df["分钟起始"].astype(str))

    demands = []
    # print(f"Generating initial demands with multiplier {volume_multiplier:.2f}...") # Reduce verbosity in loop
    for route_code, group in df.groupby("线路编码"):
        try:
            route_info = df_route_details[df_route_details["线路编码"] == route_code].iloc[0]
            group = group.sort_values("datetime")
            node_time = parse_time_robustly(route_info["发运节点"])
            base_date = route_base_dates.get(route_code)

            if node_time is None or base_date is None:
                # print(f"Warning: Missing node time or base date for route {route_code}. Skipping.") # Reduce verbosity
                continue
            node_departure_dt = datetime.datetime.combine(base_date, node_time)

            accumulated_volume = 0
            for _, row in group.iterrows():
                accumulated_volume += int(row["包裹量"])
                while accumulated_volume >= CAPACITY_CONTAINER:
                    demands.append({
                        "route_codes": [route_code],       # List to allow merging later
                        "volume": CAPACITY_CONTAINER,
                        "depart_datetime": row["datetime"], # Depart when full
                        "fleet_id": route_info["车队编码"],
                        "route_duration_hours": float(route_info["在途时长"]),
                        "variable_cost": float(route_info["自有变动成本"]),
                        "external_cost": float(route_info["外部承运商成本"]),
                        "use_container": True,           # Default to container
                        # Store stations for easier pairing check later
                        "origin_station_id": extract_station_id(route_info['起始场地']),
                        "dest_station_id": extract_station_id(route_info['目的场地']),
                    })
                    accumulated_volume -= CAPACITY_CONTAINER

            if accumulated_volume > 0:
                demands.append({
                    "route_codes": [route_code],
                    "volume": accumulated_volume,
                    "depart_datetime": node_departure_dt, # Leftovers depart at base time
                    "fleet_id": route_info["车队编码"],
                    "route_duration_hours": float(route_info["在途时长"]),
                    "variable_cost": float(route_info["自有变动成本"]),
                    "external_cost": float(route_info["外部承运商成本"]),
                    "use_container": True,
                    "origin_station_id": extract_station_id(route_info['起始场地']),
                    "dest_station_id": extract_station_id(route_info['目的场地']),
                })
        except (IndexError, KeyError, ValueError, TypeError) as e:
             # print(f"Warning: Error processing route {route_code}: {e}. Skipping.") # Reduce verbosity
             pass # Skip route silently on error
    # print(f"Generated {len(demands)} initial demands.") # Reduce verbosity
    return demands

def pair_demands_multi_stop(demands, allowed_station_pairs):
    """Attempts to combine demands into multi-stop routes (串点)."""
    if not demands: return []
    # Sort for efficient processing
    sorted_indices = sorted(range(len(demands)), key=lambda i: demands[i]["depart_datetime"])
    processed = [False] * len(demands)
    paired_demands = []
    pairing_window_delta = datetime.timedelta(minutes=PAIRING_WINDOW_MINUTES)

    # print("Attempting multi-stop pairing (串点)...") # Reduce verbosity
    for i_idx in sorted_indices:
        if processed[i_idx]: continue
        demand1 = demands[i_idx]
        # Cannot pair demands that are already full container
        if demand1["volume"] == CAPACITY_CONTAINER:
            paired_demands.append(demand1); processed[i_idx] = True; continue

        was_paired = False
        for j_idx in sorted_indices:
            if j_idx <= i_idx or processed[j_idx]: continue
            demand2 = demands[j_idx]

            # Optimization: If we've passed the time window for this fleet, break inner loop
            # Note: This assumes demands are sorted by FLEET then TIME for full optimization.
            # Current sort is only by time, so a demand from another fleet might appear earlier
            # but be too far in time. Let's keep original logic which checks abs time diff.
            # If fleets mismatch, continue checking later demands for current demand1.
            if demand1["fleet_id"] != demand2["fleet_id"]: continue # Still require same fleet

            # Only check within the time window
            time_diff = abs((demand2["depart_datetime"] - demand1["depart_datetime"]).total_seconds())
            if time_diff > pairing_window_delta.total_seconds(): continue

            # Volume check
            if demand1["volume"] + demand2["volume"] > CAPACITY_CONTAINER: continue

            # Station pair check using stored IDs
            # Collect all unique station IDs involved in the combined potential trip
            stations_d1 = {demand1.get('origin_station_id'), demand1.get('dest_station_id')}
            stations_d2 = {demand2.get('origin_station_id'), demand2.get('dest_station_id')}
            all_involved_ids = list((stations_d1 | stations_d2) - {None}) # Union, remove None

            is_pairable = True
            if len(all_involved_ids) > 1: # Need at least two distinct stations
                 # Check if ALL pairs of distinct stations among the combined set are allowed
                 for s1_id in all_involved_ids:
                      for s2_id in all_involved_ids:
                           if s1_id != s2_id:
                                pair_to_check = tuple(sorted((s1_id, s2_id)))
                                if pair_to_check not in allowed_station_pairs:
                                    is_pairable = False; break # Found an invalid pair
                      if not is_pairable: break # Stop checking pairs for this demand combination

            if not is_pairable: continue # This pairing is not allowed

            # If all checks pass, merge
            # print(f"  Pairing demand @ {demand1['depart_datetime'].strftime('%H:%M')} w/ demand @ {demand2['depart_datetime'].strftime('%H:%M')}") # Reduce verbosity
            merged = {
                "route_codes": list(set(demand1["route_codes"] + demand2["route_codes"])),
                "volume": demand1["volume"] + demand2["volume"],
                "depart_datetime": max(demand1["depart_datetime"], demand2["depart_datetime"]),
                "fleet_id": demand1["fleet_id"],
                "route_duration_hours": max(demand1["route_duration_hours"], demand2["route_duration_hours"]),
                "variable_cost": max(demand1["variable_cost"], demand1["variable_cost"]), # Corrected typo? Should be d2? Assuming max(v1, v2)
                "external_cost": max(demand1["external_cost"], demand2["external_cost"]),
                "use_container": True, # Pairing always uses container initially
                # Keep station IDs from the first demand for simplicity after merge
                "origin_station_id": demand1.get('origin_station_id'),
                "dest_station_id": demand1.get('dest_station_id'),
            }
            paired_demands.append(merged)
            processed[i_idx] = processed[j_idx] = True
            was_paired = True
            break # Demand i is paired, move to next i

        # If demand i was not paired with any subsequent demand
        if not was_paired:
            paired_demands.append(demand1)
            processed[i_idx] = True

    # print(f"Demands after multi-stop pairing: {len(paired_demands)}.") # Reduce verbosity
    return paired_demands


def merge_to_remove_container(demands_after_pairing):
    """Attempts to merge two containerized demands into non-containerized."""
    if not demands_after_pairing: return []

    # Group demand indices by fleet and 10-minute departure bucket for efficient search
    demands_by_bucket = collections.defaultdict(list)
    for i, d in enumerate(demands_after_pairing):
        bucket_time = d["depart_datetime"].replace(minute=(d["depart_datetime"].minute // PAIRING_WINDOW_MINUTES) * PAIRING_WINDOW_MINUTES, second=0, microsecond=0)
        key = (d["fleet_id"], bucket_time)
        demands_by_bucket[key].append(i)

    processed = [False] * len(demands_after_pairing)
    final_demands = []

    # print("Attempting decontainerization merge...") # Reduce verbosity
    # Iterate through original demands list to maintain order somewhat
    for i, demand1 in enumerate(demands_after_pairing):
        if processed[i]: continue

        merged_decontainerized = False
        # Only try if it's currently a container trip with potential to merge (volume < 800)
        if demand1["use_container"] and demand1["volume"] < CAPACITY_CONTAINER:
            bucket_time = demand1["depart_datetime"].replace(minute=(demand1["depart_datetime"].minute // PAIRING_WINDOW_MINUTES) * PAIRING_WINDOW_MINUTES, second=0, microsecond=0)
            key = (demand1["fleet_id"], bucket_time)
            candidate_indices = demands_by_bucket.get(key, [])

            for j in candidate_indices:
                if i == j or processed[j]: continue

                demand2 = demands_after_pairing[j]

                # Check merge conditions
                if (demand2["use_container"] and # Both must be containerized
                    demand1["volume"] + demand2["volume"] <= CAPACITY_NORMAL): # Combined volume <= 1000

                    # print(f"  Decontainerizing & merging demands @ {demand1['depart_datetime'].strftime('%H:%M')} & {demand2['depart_datetime'].strftime('%H:%M')}") # Reduce verbosity
                    merged = {
                        "route_codes": list(set(demand1["route_codes"] + demand2["route_codes"])),
                        "volume": demand1["volume"] + demand2["volume"],
                        "depart_datetime": max(demand1["depart_datetime"], demand2["depart_datetime"]),
                        "fleet_id": demand1["fleet_id"],
                        "route_duration_hours": max(demand1["route_duration_hours"], demand2["route_duration_hours"]),
                        "variable_cost": max(demand1["variable_cost"], demand2["variable_cost"]),
                        "external_cost": max(demand1["external_cost"], demand2["external_cost"]),
                        "use_container": False, # Merged trip does NOT use container
                        "origin_station_id": demand1.get('origin_station_id'),
                        "dest_station_id": demand1.get('dest_station_id'),
                    }
                    final_demands.append(merged)
                    processed[i] = processed[j] = True
                    merged_decontainerized = True
                    break # Stop searching for demand1

        # If demand1 wasn't merged, add it as is
        if not merged_decontainerized:
            final_demands.append(demand1)
            processed[i] = True

    # print(f"Demands after decontainerization attempts: {len(final_demands)}.") # Reduce verbosity
    return final_demands


def assign_vehicles_to_demands(final_demands, fleet_capacities):
    """Assigns vehicles (internal or external) to demands using a greedy approach."""
    if not final_demands: return []

    # Track vehicle availability: {fleet_id: [{'id': vehicle_id, 'return_time': datetime}, ...]}
    vehicle_pool = {f: [] for f in fleet_capacities}
    vehicle_sequence_num = {f: 1 for f in fleet_capacities} # For naming new vehicles
    assignments = [] # List to store assignment tuples: (demand_dict, assigned_vehicle_id, is_external_bool)

    print("Assigning vehicles...")
    # Process demands sorted by departure time
    for demand in sorted(final_demands, key=lambda x: x["depart_datetime"]):
        fleet_id = demand["fleet_id"]
        departure_dt = demand["depart_datetime"]
        assigned_vehicle_id = None
        is_external = False
        demand_to_assign = demand.copy() # Work with a copy in case we modify use_container

        # 1. Find earliest available owned vehicle in the pool
        # Filter to find vehicles available by departure time
        available_vehicles = [v for v in vehicle_pool.get(fleet_id, []) if v['return_time'] <= departure_dt]

        best_vehicle = None
        if available_vehicles:
             # Choose the one returning earliest if multiple are free
            best_vehicle = min(available_vehicles, key=lambda v: v['return_time'])

        if best_vehicle:
            assigned_vehicle_id = best_vehicle['id']
            best_vehicle['return_time'] = departure_dt + calculate_turnaround_duration(demand_to_assign) # Update return time
            # print(f"  Reusing vehicle {assigned_vehicle_id}...") # Reduce verbosity
        # 2. Allocate new owned vehicle if fleet capacity allows
        elif fleet_id in fleet_capacities and len(vehicle_pool.get(fleet_id, [])) < fleet_capacities[fleet_id]:
            new_id = f"{fleet_id}-V{vehicle_sequence_num[fleet_id]:02d}"; vehicle_sequence_num[fleet_id] += 1
            new_vehicle_record = {'id': new_id, 'return_time': departure_dt + calculate_turnaround_duration(demand_to_assign)}
            vehicle_pool.setdefault(fleet_id, []).append(new_vehicle_record)
            assigned_vehicle_id = new_id
            # print(f"  Allocating new vehicle {assigned_vehicle_id}...") # Reduce verbosity
        # 3. Try dynamic container switch (if currently non-containerized but could be containerized)
        elif not demand_to_assign['use_container'] and demand_to_assign['volume'] <= CAPACITY_CONTAINER:
            # Tentatively switch to container to calculate potentially earlier return time
            demand_with_container = demand_to_assign.copy(); demand_with_container['use_container'] = True
            earlier_return_duration = calculate_turnaround_duration(demand_with_container)
            tentative_return_time = departure_dt + earlier_return_duration

            # Check if switching makes an *existing* vehicle available by the *original* departure time
            # The condition should still be availability *at the original departure time*
            available_after_switch = [v for v in vehicle_pool.get(fleet_id, []) if v['return_time'] <= departure_dt]
            best_after_switch = min(available_after_switch, key=lambda v: v['return_time']) if available_after_switch else None

            if best_after_switch:
                 # Assign the vehicle using the containerized version of the demand
                 assigned_vehicle_id = best_after_switch['id']
                 best_after_switch['return_time'] = tentative_return_time # Use the earlier return time
                 demand_to_assign = demand_with_container # Commit to using the container for this assignment
                 # print(f"  Dynamic Switch: Reusing {assigned_vehicle_id}...") # Reduce verbosity
            else:
                 # Switch didn't help, assign original demand externally
                 assigned_vehicle_id = EXTERNAL_VEHICLE_ID; is_external = True
                 # print(f"  Assigning EXTERNAL (switch ineffective)...") # Reduce verbosity
        # 4. Assign externally if no other option
        else:
             assigned_vehicle_id = EXTERNAL_VEHICLE_ID; is_external = True
             # print(f"  Assigning EXTERNAL...") # Reduce verbosity

        assignments.append((demand_to_assign, assigned_vehicle_id, is_external))

        # Sort vehicle pool by return time for faster lookup next time
        # This part needs the specific vehicle object reference, not just the ID
        if fleet_id in vehicle_pool:
            vehicle_pool[fleet_id].sort(key=lambda v: v['return_time'])

    print(f"Vehicle assignment complete. Total assignments: {len(assignments)}")
    return assignments

def calculate_simulation_metrics(assignments):
    """Calculates summary metrics based on the vehicle assignments."""
    if not assignments:
        return {"外部车辆数": 0, "自有车周转次数": 0, "车辆均包裹量": 0, "总成本": 0}

    external_count = sum(1 for _, _, is_ext in assignments if is_ext)
    owned_vehicle_trips_counter = collections.Counter() # Count trips per specific *owned* vehicle ID
    total_volume = 0
    total_owned_variable_cost = 0
    used_owned_vehicle_ids = set() # Track unique owned vehicles used

    for demand_info, vehicle_id, is_external in assignments:
        total_volume += demand_info['volume']
        if not is_external:
             # Owned vehicle, vehicle_id is like "Fleet-V01"
             owned_vehicle_trips_counter[vehicle_id] += 1
             used_owned_vehicle_ids.add(vehicle_id)
             total_owned_variable_cost += demand_info['variable_cost'] # Variable cost for owned trips
        # else: external cost is handled below

    num_owned_vehicles_used = len(used_owned_vehicle_ids)
    total_owned_trips = sum(owned_vehicle_trips_counter.values())

    # Calculate turnover rate = total owned trips / number of unique owned vehicles used
    avg_turnover = total_owned_trips / num_owned_vehicles_used if num_owned_vehicles_used > 0 else 0

    # Calculate total dispatch units = unique owned vehicles used + external trips
    total_dispatch_units = num_owned_vehicles_used + external_count
    avg_packages_per_vehicle = total_volume / total_dispatch_units if total_dispatch_units > 0 else 0

    # Calculate total cost
    total_fixed_cost = num_owned_vehicles_used * FIXED_COST_PER_OWNED_VEHICLE
    total_external_cost = external_count * ASSUMED_EXTERNAL_COST_PER_TRIP
    total_cost = total_fixed_cost + total_owned_variable_cost + total_external_cost

    return {
        "外部车辆数": external_count, "自有车周转次数": avg_turnover,
        "车辆均包裹量": avg_packages_per_vehicle, "总成本": total_cost,
    }

# ---------------- Main Execution Workflow ----------------

# 1. Load Base Data (already loaded globally as in original)
# print("--- Loading Base Data ---") # Reduce verbosity as already done above
# try:
#     df_daily_schedule = pd.read_excel(DAILY_SCHEDULE_FILE)
#     df_ten_min_predictions = pd.read_excel(TEN_MIN_PREDICTIONS_FILE)
#     df_station_pairs = pd.read_excel(STATION_PAIRS_FILE)
#     df_fleet_info = pd.read_excel(FLEET_INFO_FILE)
#     df_route_info = pd.read_excel(ROUTE_INFO_FILE)
#     print("Base data loaded successfully.")
# except Exception as e:
#     print(f"Fatal Error loading base data: {e}")
#     exit()

# # Pre-process global data (already done above)
# fleet_capacities = df_fleet_info.set_index("车队编码")["自有车数量"].to_dict()
# route_base_dates = {r["线路编码"]: pd.to_datetime(r["日期"]).date() for _, r in df_daily_schedule.iterrows()}
# allowed_station_pairs_ids = set()
# for _, row in df_station_pairs.iterrows():
#     s1_id = extract_station_id(str(row.iloc[0]))
#     s2_id = extract_station_id(str(row.iloc[1]))
#     if s1_id is not None and s2_id is not None:
#         allowed_station_pairs_ids.add(tuple(sorted((s1_id, s2_id))))
#         allowed_station_pairs_ids.add((s2_id, s1_id))

print("--- Running Simulation Scenarios ---")

# 2. Define Scenarios
scenarios = {
    "-30%": 0.7, "-20%": 0.8, "-10%": 0.9,
    "基准":  1.0,
    "+10%": 1.1, "+20%": 1.2, "+30%": 1.3,
}
scenario_results = {}

# 3. Run Simulation for Each Scenario
for scenario_name, multiplier in scenarios.items():
    print(f"\n--- Running Scenario: {scenario_name} (Multiplier: {multiplier:.1f}) ---")
    # Stage 1: Generate initial demands
    current_demands = generate_initial_demands(
        df_ten_min_predictions, df_route_info, route_base_dates, multiplier
    )
    # Stage 2: Pair demands (multi-stop)
    current_demands = pair_demands_multi_stop(
        current_demands, allowed_station_pairs_ids # Pass allowed pair IDs
    )
    # Stage 3: Decontainerize and merge
    current_demands = merge_to_remove_container(current_demands) # Does not need allowed_station_pairs_ids
    # Stage 4: Assign vehicles
    assignments = assign_vehicles_to_demands(current_demands, fleet_capacities)
    # Stage 5: Calculate metrics for the scenario
    scenario_results[scenario_name] = calculate_simulation_metrics(assignments) # Metrics based on assignments
    gc.collect() # Clean up memory between scenarios

# 4. Format and Display Results
print("\n--- Simulation Results ---")
df_results_final = pd.DataFrame(scenario_results).T
# Rename columns for final display to match original request
df_results_final.columns = ['外部车辆数', '自有车周转次数', '车辆均包裹量', '总成本']
print(df_results_final)

# 5. Plot Results
print("\n--- Plotting Results ---")
# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

fig, axs = plt.subplots(len(df_results_final.columns), 1, figsize=(8, 12), sharex=True)
if len(df_results_final.columns) == 1: axs = [axs] # Ensure axs is iterable

for i, col_name in enumerate(df_results_final.columns):
    bars = axs[i].bar(df_results_final.index, df_results_final[col_name], color=plt.cm.viridis(i/len(df_results_final.columns)))
    axs[i].set_title(col_name, fontsize=12)
    axs[i].set_ylabel("") # Keep y-label empty as title describes metric
    axs[i].tick_params(axis='y', labelsize=9)
    # Add data labels on bars
    try: # Use try-except for bar_label in case of matplotlib version issues
        axs[i].bar_label(bars, fmt='{:,.0f}' if col_name == '外部车辆数' or col_name == '总成本' else '{:,.2f}', fontsize=8, padding=3)
    except Exception as e:
        # print(f"Warning: Could not add bar labels for '{col_name}': {e}") # Reduce verbosity
        pass # Silently pass on labeling errors

# Set common xlabel at the bottom
axs[-1].set_xlabel("预测场景", fontsize=11)
axs[-1].tick_params(axis='x', rotation=0, labelsize=10)

fig.suptitle("不同预测场景下的调度指标对比", fontsize=14, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to make space for suptitle

# Save the plot
try:
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"Results plot saved to {OUTPUT_PLOT_FILE}")
except Exception as e:
    print(f"Error saving results plot: {e}")

plt.show()

print("\n--- Script Finished ---")