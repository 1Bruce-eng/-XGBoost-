# START OF FILE 时间序列分析.py

# 时间序列分析及预测脚本 (简化版 - 路线特定生产时间)

# 作者：XXX
# 日期：XXXX-XX-XX (修改日期以反映变更)

# --------------------------
# 1. 导入必要库
# --------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import numpy as np
from tqdm import tqdm
import math
import os
import gc

# 配置 matplotlib 及忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# --------------------------
# 2. 配置参数
# --------------------------
INPUT_FILE = './附件/附件2.xlsx'
# Updated output filenames
OUTPUT_FILE_10MIN = '预测结果_10分钟.xlsx'
OUTPUT_FILE_DAILY = '预测结果_天粒度.xlsx'
PLOT_OUTPUT_DIR = '预测评估图表' # Updated plot dir

ROUTES_TO_PLOT_AND_EVALUATE = [
    '场地3 - 站点83 - 1400',
    '场地3 - 站点83 - 0600'
]

# 模型参数
SEASONAL_PERIOD = 6
MODEL_LAG = 9
XGB_PARAMS = { # Using params from file where Feature Engineering was removed
    'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
    'objective': 'reg:squarederror'
}
np.random.seed(42)

# 训练/评估参数
MIN_RECORDS_FOR_TRAINING = MODEL_LAG + 10
EVAL_TEST_SIZE_RECORDS = 24 * 7

# --------------------------
# 3. 辅助函数
# --------------------------
def create_lag_features(data_array, lag_order):
    """生成滞后特征"""
    if len(data_array) <= lag_order: return np.array([]), np.array([])
    X, y = [], []
    for i in range(lag_order, len(data_array)):
        X.append(data_array[i - lag_order:i])
        y.append(data_array[i])
    return np.array(X), np.array(y)

def get_route_production_hours(route_code):
    """获取路线的生产小时列表"""
    if route_code.endswith(' - 1400'): return list(range(11, 14))
    if route_code.endswith(' - 0600'): return list(range(0, 6)) + list(range(21, 24))
    all_hours = set(range(24)); non_prod = set(range(6, 11)) | set(range(14, 21))
    return sorted(list(all_hours - non_prod))

def plot_predictions(route_code, hist_df, test_preds_df, future_preds_df, metrics, output_dir, prod_hours):
    """绘制历史和预测数据 (输入为已过滤的生产时段数据)"""
    # (Plotting function remains the same as previous version)
    fig, ax = plt.subplots(figsize=(18, 7))
    hist_df = hist_df if hist_df is not None else pd.DataFrame(columns=['小时开始时间', '总包裹量', 'split'])
    test_preds_df = test_preds_df if test_preds_df is not None else pd.DataFrame(columns=['小时开始时间', '预测包裹量'])
    future_preds_df = future_preds_df if future_preds_df is not None else pd.DataFrame(columns=['小时开始时间', '预测包裹量'])
    if hist_df.empty and future_preds_df.empty: print(f"  线路 {route_code}: 无生产时段数据绘图。"); plt.close(fig); return
    last_hist_time = pd.NaT if hist_df.empty else hist_df['小时开始时间'].max()
    hourly_thresh = pd.Timedelta(hours=1)
    def plot_segments(data, y_col, color, label, marker='.', linestyle='-'):
        if data.empty: return False
        plotted = False; times, values = [], []
        data = data.sort_values('小时开始时间')
        for i in range(len(data)):
            t, v = data.iloc[i]['小时开始时间'], data.iloc[i][y_col]
            if pd.isna(t): continue
            if i > 0:
                prev_t = data.iloc[i-1]['小时开始时间']
                if pd.isna(prev_t) or (pd.to_datetime(t) - pd.to_datetime(prev_t) > hourly_thresh):
                    if times: ax.plot(times, values, label=(label if not plotted else None), color=color, alpha=0.7, marker=marker, linestyle=linestyle); plotted = True
                    times, values = [t], [v]
                else: times.append(t); values.append(v)
            else: times, values = [t], [v]
        if times: ax.plot(times, values, label=(label if not plotted else None), color=color, alpha=0.7, marker=marker, linestyle=linestyle); plotted = True
        return plotted
    if not hist_df.empty:
        plot_segments(hist_df[hist_df['split'] == 'train'], '总包裹量', 'blue', '历史训练 (生产)')
        plot_segments(hist_df[hist_df['split'] == 'test'], '总包裹量', 'orange', '历史测试 (生产)')
    if not test_preds_df.empty: ax.plot(test_preds_df['小时开始时间'], test_preds_df['预测包裹量'], label='测试预测 (生产)', color='green', alpha=0.9, marker='x', linestyle='--')
    if not future_preds_df.empty:
        if not pd.isna(last_hist_time):
            last_hist_val_series = hist_df.loc[hist_df['小时开始时间'] == last_hist_time, '总包裹量']
            if not last_hist_val_series.empty:
                last_hist_val = last_hist_val_series.iloc[0]; first_pred_t = future_preds_df['小时开始时间'].min()
                if pd.to_datetime(first_pred_t) - pd.to_datetime(last_hist_time) <= hourly_thresh:
                     times = pd.concat([pd.Series([last_hist_time]), future_preds_df['小时开始时间']]).reset_index(drop=True)
                     vals = pd.concat([pd.Series([last_hist_val]), future_preds_df['预测包裹量']]).reset_index(drop=True)
                     ax.plot(times, vals, label='未来预测', color='red', alpha=0.8, marker='^', linestyle=':') # Removed "(生产)" label
                else: ax.plot(future_preds_df['小时开始时间'], future_preds_df['预测包裹量'], label='未来预测', color='red', alpha=0.8, marker='^', linestyle=':') # Removed "(生产)" label
            else: ax.plot(future_preds_df['小时开始时间'], future_preds_df['预测包裹量'], label='未来预测', color='red', alpha=0.8, marker='^', linestyle=':') # Removed "(生产)" label
        else: ax.plot(future_preds_df['小时开始时间'], future_preds_df['预测包裹量'], label='未来预测', color='red', alpha=0.8, marker='^', linestyle=':') # Removed "(生产)" label
    # Updated title to reflect fixed prediction start
    title = f"线路: {route_code} (训练生产时段: {prod_hours}) - 预测自 {future_preds_df['小时开始时间'].min().strftime('%Y-%m-%d %H:%M') if not future_preds_df.empty else 'N/A'}"
    ax.set_title(title); ax.set_xlabel("时间"); ax.set_ylabel("总包裹量 (小时)")
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    try:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=20)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=30, ha='right')
    except Exception as e: print(f"警告: 格式化X轴失败: {e}"); plt.xticks(rotation=30, ha='right')
    if metrics:
        txt = f"评估指标 (基于原始模型预测):\nTrain R^2: {metrics.get('Train R2', 'N/A'):.3f}\nTest R^2: {metrics.get('Test R2', 'N/A'):.3f}"
        plt.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout(); fname = os.path.join(output_dir, f"{route_code.replace(' ', '_').replace('/', '-')}_预测图_固定开始时间.png") # Updated filename
    try: plt.savefig(fname); print(f"  图表保存至: {fname}")
    except Exception as e: print(f"  错误: 保存图表失败 {fname}: {e}")
    plt.close(fig)


# --------------------------
# 4. 数据加载与预处理 (整体)
# --------------------------
print(f"加载数据: {INPUT_FILE}")
try: raw_data_base = pd.read_excel(INPUT_FILE, usecols=['线路编码', '日期', '分钟起始', '包裹量'])
except Exception as e: print(f"错误: 无法加载 {INPUT_FILE}: {e}"); exit()
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
print("确定年份...")
try:
    _time_check = pd.to_datetime(raw_data_base.head(1000)["日期"].astype(str) + " " + raw_data_base.head(1000)["分钟起始"].astype(str), errors='coerce').dropna()
    if _time_check.empty: _time_check = pd.to_datetime(raw_data_base["日期"].astype(str) + " " + raw_data_base["分钟起始"].astype(str), errors='coerce').dropna()
    if _time_check.empty: raise ValueError("无有效时间数据")
    YEAR = _time_check.max().year
    # NEW: Define the fixed prediction start time for all routes
    FIXED_PREDICTION_START_TS = pd.Timestamp(f"{YEAR}-12-15 21:00:00")
    TARGET_PREDICTION_END = pd.Timestamp(f"{YEAR}-12-16 14:00:00")

    # These variables are related to the OLD 0600 rule and filtering output, keep for filtering output
    # FIXED_START_TS_0600 = pd.Timestamp(f"{YEAR}-12-15 00:00:00") # No longer used for prediction start
    RETENTION_START_TS_0600 = pd.Timestamp(f"{YEAR}-12-15 21:00:00") # Used for filtering output for 0600

    print(f"年份: {YEAR}")
    print(f"所有线路预测将从固定时间 {FIXED_PREDICTION_START_TS} 开始生成，直到 {TARGET_PREDICTION_END}") # Updated message
    print(f"特殊规则: -0600 路线输出仍然从 {RETENTION_START_TS_0600} 开始过滤") # Updated message
    del _time_check; gc.collect()
except Exception as e: print(f"错误: 确定年份失败: {e}"); exit()
all_route_codes = raw_data_base['线路编码'].unique(); print(f"找到 {len(all_route_codes)} 条线路。")

# --------------------------
# 5. 逐条线路处理与预测
# --------------------------
all_preds_10min = []; all_preds_daily = []

# Updated TQDM desc
for route_code in tqdm(all_route_codes, desc="处理线路 (固定预测开始时间)"):
    try:
        route_data = raw_data_base[raw_data_base['线路编码'] == route_code].copy();
        if route_data.empty: continue

        # --- 预处理和小时聚合 ---
        route_data['包裹量'] = route_data['包裹量'].fillna(0).astype(int)
        route_data['时间'] = pd.to_datetime(route_data["日期"].astype(str) + " " + route_data["分钟起始"].astype(str), errors='coerce')
        route_data = route_data.dropna(subset=['时间'])
        if route_data.empty: continue
        valid_time_data = route_data.set_index('时间')[['包裹量']].sort_index()
        try:
            hourly_agg = valid_time_data['包裹量'].resample('H').sum().reset_index()
            orig_stamps = valid_time_data.index.floor('H').unique()
            hourly_agg = hourly_agg[hourly_agg['时间'].isin(orig_stamps)].rename(columns={'时间': '小时开始时间', '包裹量': '总包裹量'})
        except Exception as e: print(f"线路 {route_code}: 聚合失败: {e}"); continue

        # --- 按路线过滤生产小时 (用于训练/评估) ---
        production_hours = get_route_production_hours(route_code)
        # Only use production hours for training data
        prod_hourly_hist = hourly_agg[hourly_agg['小时开始时间'].dt.hour.isin(production_hours)].copy()
        prod_hourly_hist = prod_hourly_hist.sort_values('小时开始时间').reset_index(drop=True)

        if len(prod_hourly_hist) < MIN_RECORDS_FOR_TRAINING:
            print(f"  线路 {route_code}: 生产时段历史数据不足 ({len(prod_hourly_hist)} 小时) 无法训练模型 (需要 {MIN_RECORDS_FOR_TRAINING} 小时). 跳过.")
            continue

        prod_hourly_values = prod_hourly_hist["总包裹量"].values
        # prod_hourly_timestamps = prod_hourly_hist["小时开始时间"].values # Not directly used for lag features anymore

        # --- 模型评估 (可选, 在原始模型上) ---
        # Eval is still based on the training data subset (production hours)
        eval_metrics = None; prod_test_preds_df = None; test_timestamps = None
        if route_code in ROUTES_TO_PLOT_AND_EVALUATE and len(prod_hourly_values) >= MODEL_LAG + EVAL_TEST_SIZE_RECORDS:
            X_eval, y_eval = create_lag_features(prod_hourly_values, MODEL_LAG)
            # Timestamps for evaluation correspond to the y_eval points in prod_hourly_hist
            ts_eval = prod_hourly_hist["小时开始时间"].iloc[MODEL_LAG:].values
            if X_eval.size > 0 and len(y_eval) > EVAL_TEST_SIZE_RECORDS:
                split_idx = len(y_eval) - EVAL_TEST_SIZE_RECORDS
                if split_idx >= 1:
                    X_tr, y_tr = X_eval[:split_idx], y_eval[:split_idx]; X_te, y_te = X_eval[split_idx:], y_eval[split_idx:]
                    test_timestamps = ts_eval[split_idx:] # Get corresponding test timestamps
                    try:
                        eval_model = XGBRegressor(**XGB_PARAMS).fit(X_tr, y_tr, verbose=False)
                        tr_pred = np.maximum(0, eval_model.predict(X_tr)); te_pred = np.maximum(0, eval_model.predict(X_te))
                        eval_metrics = {'Train R2': r2_score(y_tr, tr_pred), 'Test R2': r2_score(y_te, te_pred)}
                        print(f"  评估 {route_code}: Train R^2={eval_metrics['Train R2']:.3f}, Test R^2={eval_metrics['Test R2']:.3f}")
                        prod_test_preds_df = pd.DataFrame({'小时开始时间': test_timestamps, '预测包裹量': te_pred})
                    except Exception as e: print(f"  线路 {route_code} 评估失败: {e}"); eval_metrics=None; prod_test_preds_df=None

        # --- 训练最终模型 ---
        X_all, y_all = create_lag_features(prod_hourly_values, MODEL_LAG)
        if X_all.size == 0:
             print(f"  线路 {route_code}: 生产时段历史数据不足以创建滞后特征 (需要至少 {MODEL_LAG+1} 小时). 跳过训练.")
             continue
        try: final_model = XGBRegressor(**XGB_PARAMS).fit(X_all, y_all, verbose=False)
        except Exception as e: print(f"  线路 {route_code} 最终模型训练失败: {e}"); continue

        # --- 预测未来小时 ---
        pred_start_ts = FIXED_PREDICTION_START_TS # NEW: Fixed prediction start time for ALL routes
        if pred_start_ts >= TARGET_PREDICTION_END:
             print(f"  线路 {route_code}: 预测开始时间 {pred_start_ts} 晚于或等于预测结束时间 {TARGET_PREDICTION_END}. 跳过预测.")
             continue

        # --- 准备预测的初始窗口 (使用所有小时的历史数据) ---
        # Find historical hourly data points *before* the fixed prediction start time
        history_for_initial_window = hourly_agg[hourly_agg['小时开始时间'] < pred_start_ts].copy()
        history_for_initial_window = history_for_initial_window.sort_values('小时开始时间')

        if len(history_for_initial_window) < MODEL_LAG:
            print(f"  线路 {route_code}: 历史数据不足 ({len(history_for_initial_window)} 小时) 无法从 {pred_start_ts} 开始预测 (需要 {MODEL_LAG} 小时). 跳过.")
            continue # Skip this route if not enough history

        # Take the last MODEL_LAG actual historical values as the initial window
        last_window = history_for_initial_window['总包裹量'].tail(MODEL_LAG).tolist()
        current_input = np.array(last_window).reshape(1, -1)

        secs_to_pred = (TARGET_PREDICTION_END - pred_start_ts).total_seconds();
        if secs_to_pred <= 0:
             print(f"  线路 {route_code}: 预测窗口时长为0或负数. 跳过预测.")
             continue
        n_preds = math.ceil(secs_to_pred / 3600)
        if n_preds <= 0:
             print(f"  线路 {route_code}: 计算得出的预测步数为0. 跳过预测.")
             continue


        future_preds_list = []; future_ts_list = []
        current_ts = pred_start_ts
        for _ in range(n_preds):
            # Stop if we reach or exceed the target end time
            if current_ts >= TARGET_PREDICTION_END:
                break

            pred = max(0, float(final_model.predict(current_input)[0]))
            future_preds_list.append(pred)
            future_ts_list.append(current_ts)

            # Prepare input for the next step
            current_input = np.roll(current_input, -1) # Shift window
            current_input[0, -1] = pred # Add the new prediction to the end
            current_ts += pd.Timedelta(hours=1) # Move to the next hour

        if not future_preds_list:
             print(f"  线路 {route_code}: 未生成任何未来小时预测值. 跳过.")
             continue

        future_preds_df = pd.DataFrame({'小时开始时间': future_ts_list, '预测包裹量': future_preds_list}).dropna()

        if future_preds_df.empty:
            print(f"  线路 {route_code}: 生成的未来小时预测 DataFrame 为空. 跳过.")
            continue

        # --- 5.3a: Removed the 0600 Swap Logic as prediction starts at 21:00 ---
        # The 0600 swap logic (0,1,2 <-> 21,22,23) is not applicable if prediction
        # starts at 21:00, as hours 0, 1, 2 will not be in the prediction range.

        # --- 5.3b: 过滤未来预测至生产小时 (虽然预测已从21点开始,但保留此步骤以匹配原始逻辑结构和只输出生产时段的需求) ---
        # For 0600 routes, this will effectively keep 21,22,23 and 0,1,2,3,4,5 from the *next* day if prediction window extends.
        # For other routes, it filters to their production hours within the prediction window.
        prod_future_hourly_preds = future_preds_df[future_preds_df['小时开始时间'].dt.hour.isin(production_hours)].copy()

        # --- 5.4: 分解至10分钟 (仅过滤后的生产小时) ---
        preds_10min_list = []; timestamps_10min_list = []
        # Iterate only over the filtered production hours for 10-min disaggregation
        for _, row in prod_future_hourly_preds.iterrows():
            hr_ts, hr_pred = row['小时开始时间'], row['预测包裹量']
            # Use a simple seasonal pattern for disaggregation (can be improved)
            # Example: peaks at 10, 30, 50 mins, lower at 0, 20, 40 mins
            # Simple fixed proportions (summing to 1) or learned proportions could be used
            # For now, keep the random approach from the original script
            props = np.random.rand(SEASONAL_PERIOD) + 1e-9; props /= props.sum() # Retain original random split

            for i in range(SEASONAL_PERIOD): # Create all 6 slots for this hour
                ts_10m = hr_ts + pd.Timedelta(minutes=10 * i)
                pred_10m = max(0, int(round(hr_pred * props[i])))
                timestamps_10min_list.append(ts_10m); preds_10min_list.append(pred_10m)

        route_10min_df = pd.DataFrame({'线路编码': route_code, '时间': timestamps_10min_list, '预测包裹量': preds_10min_list}).dropna()

        # --- 5.4b: 过滤至目标输出窗口 ---
        # Filter 10-min predictions to the final desired output window
        route_10min_df_target = route_10min_df[(route_10min_df['时间'] >= FIXED_PREDICTION_START_TS) & (route_10min_df['时间'] < TARGET_PREDICTION_END)].copy()

        # Specific filtering for -0600 routes output start time
        if route_code.endswith(' - 0600'):
             # For 0600 routes, ensure output starts *exactly* from RETENTION_START_TS_0600
            route_10min_df_target = route_10min_df_target[route_10min_df_target['时间'] >= RETENTION_START_TS_0600].copy()


        # --- 5.4c: 收集结果 ---
        if not route_10min_df_target.empty:
            all_preds_10min.append(route_10min_df_target)
            try:
                # Aggregate the TARGET filtered 10-min data to daily
                route_10min_df_target['预测日期'] = route_10min_df_target['时间'].dt.date
                daily = route_10min_df_target.groupby('预测日期')['预测包裹量'].sum().reset_index()
                daily['线路编码'] = route_code
                all_preds_daily.append(daily[['线路编码', '预测日期', '预测包裹量']].rename(columns={'预测包裹量': '预测总包裹量'}))
            except Exception as e: print(f"  线路 {route_code}: 日聚合失败: {e}")
        else:
             print(f"  线路 {route_code}: 在输出窗口 [{FIXED_PREDICTION_START_TS}, {TARGET_PREDICTION_END}) 内无预测结果.")
             if route_code.endswith(' - 0600'):
                  print(f"  (0600路线额外过滤: >= {RETENTION_START_TS_0600})")


        # --- 5.5: 绘图 (如果需要) ---
        if route_code in ROUTES_TO_PLOT_AND_EVALUATE:
            hist_plot = None
            if prod_hourly_hist is not None and not prod_hourly_hist.empty:
                 hist_plot = prod_hourly_hist.copy()
                 hist_plot['split'] = 'train'
                 if test_timestamps is not None: hist_plot.loc[hist_plot['小时开始时间'].isin(test_timestamps), 'split'] = 'test'

            plot_future_preds = None
            # Plot future predictions (already filtered to production hours by prod_future_hourly_preds)
            if prod_future_hourly_preds is not None and not prod_future_hourly_preds.empty:
                 plot_future_preds = prod_future_hourly_preds.copy()
                 # For 0600 routes, plot only from the retention start time
                 if route_code.endswith(' - 0600'):
                     plot_future_preds = plot_future_preds[plot_future_preds['小时开始时间'] >= RETENTION_START_TS_0600].copy()
                 if plot_future_preds.empty: plot_future_preds = None # Set to None if filter results in empty df


            # Pass data (future predictions now start from FIXED_PREDICTION_START_TS and potentially filtered for plot)
            # Note: eval_metrics are still based on the original model trained on prod_hourly_hist
            plot_predictions(route_code, hist_plot, prod_test_preds_df, plot_future_preds, eval_metrics, PLOT_OUTPUT_DIR, production_hours)

        # --- 清理内存 ---
        del route_data, valid_time_data, hourly_agg, prod_hourly_hist, prod_hourly_values # prod_hourly_timestamps
        if 'history_for_initial_window' in locals(): del history_for_initial_window
        if 'X_all' in locals(): del X_all, y_all
        if 'final_model' in locals(): del final_model
        if 'X_eval' in locals(): del X_eval, y_eval, X_tr, y_tr, X_te, y_te
        if 'eval_model' in locals(): del eval_model
        if 'test_timestamps' in locals(): test_timestamps = None
        if 'prod_test_preds_df' in locals(): prod_test_preds_df = None
        if 'future_preds_df' in locals(): del future_preds_df
        if 'prod_future_hourly_preds' in locals(): del prod_future_hourly_preds
        if 'route_10min_df' in locals(): del route_10min_df, route_10min_df_target
        if 'hist_plot' in locals(): del hist_plot
        if 'plot_future_preds' in locals(): plot_future_preds = None
        gc.collect()

    except MemoryError: print(f"内存错误: {route_code}"); gc.collect(); continue
    except Exception as e: print(f"处理 {route_code} 出错: {e}"); import traceback; traceback.print_exc(); gc.collect(); continue

del raw_data_base; gc.collect()

# --------------------------
# 6. 合并与输出结果
# --------------------------
print("\n合并与保存结果...")
if all_preds_10min:
    try:
        final_10min = pd.concat(all_preds_10min, ignore_index=True); final_10min['时间'] = pd.to_datetime(final_10min['时间'])
        final_10min['预测包裹量'] = final_10min['预测包裹量'].astype(int); final_10min = final_10min.sort_values(by=['线路编码', '时间']).reset_index(drop=True)
        print(f"生成10分钟预测表 ({len(final_10min)} 条)...")
        final_10min[['线路编码', '时间', '预测包裹量']].to_excel(OUTPUT_FILE_10MIN, index=False, engine='openpyxl'); print(f"保存至: {OUTPUT_FILE_10MIN}")
        del final_10min; gc.collect()
    except Exception as e: print(f"错误: 保存10分钟结果失败: {e}")
else: print("警告: 无10分钟预测结果。")
if all_preds_daily:
    try:
        final_daily = pd.concat(all_preds_daily, ignore_index=True); final_daily['预测日期'] = pd.to_datetime(final_daily['预测日期']).dt.date
        final_daily['预测总包裹量'] = final_daily['预测总包裹量'].astype(int); final_daily = final_daily.sort_values(by=['线路编码', '预测日期']).reset_index(drop=True)
        print(f"生成天粒度预测表 ({len(final_daily)} 条)...")
        final_daily[['线路编码', '预测日期', '预测总包裹量']].to_excel(OUTPUT_FILE_DAILY, index=False, engine='openpyxl'); print(f"保存至: {OUTPUT_FILE_DAILY}")
        del final_daily; gc.collect()
    except Exception as e: print(f"错误: 保存天粒度结果失败: {e}")
else: print("警告: 无天粒度预测结果。")
print("\n--- 处理完成 ---")

# END OF FILE