#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：雷射掃描 → AdaBoost 分類 → 離散型 Bayes Filter (DBF) 追蹤與機率估計。
"""

import math
import numpy as np
import rospy
# ROS 訊息類型
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
# Rviz 視覺化 Marker 相關
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point 
import time

# === 1. MATLAB AdaBoost 模型參數 ===================================
# --- BOX 模型 (Class ID: 2) ---
T_BOX = 50 
ALPHA_BOX = np.array([1.3965, 0.7808, 1.0233, 0.5580, 0.3930, 0.4301, 0.5150, 0.4107, 0.3595, 0.4706, 0.4811, 0.4211, 0.3999, 0.3484, 0.3298, 0.2847, 0.2768, 0.3305, 0.2624, 0.3833, 0.2913, 0.2644, 0.2864, 0.2559, 0.3792, 0.2782, 0.2654, 0.2309, 0.2383, 0.2415, 0.2570, 0.1987, 0.2199, 0.2416, 0.2227, 0.2203, 0.1884, 0.1993, 0.2813, 0.2013, 0.1780, 0.2194, 0.2273, 0.2230, 0.1804, 0.1830, 0.1863, 0.1743, 0.1607, 0.1600], dtype=float) 
STUMPS_BOX = [[3, 0.0134, -1], [2, 0.0868, -1], [1, 26.5000, 1], [4, 0.0002, -1], [1, 36.5000, 1], [4, 0.4170, -1], [2, 2.3475, 1], [1, 21.5000, 1], [4, 0.0144, -1], [3, 1.4384, 1], [4, 1.4951, -1], [2, 0.0185, -1], [1, 38.5000, 1], [4, 0.6019, -1], [2, 1.9452, 1], [4, 1.4951, -1], [3, 0.2151, 1], [2, 0.0185, -1], [1, 11.5000, 1], [4, 0.0046, -1], [3, 1.5311, 1], [4, 6.3139, -1], [2, 2.3475, 1], [2, 0.3145, -1], [1, 15.5000, 1], [2, 0.1667, -1], [3, 1.0235, 1], [4, 0.2565, -1], [1, 54.5000, 1], [2, 0.0185, -1], [4, 1.4951, -1], [2, 1.9452, 1], [1, 6.5000, 1], [2, 0.1667, -1], [3, 1.0235, 1], [5, 1.7467, -1], [2, 2.3475, 1], [1, 15.5000, 1], [4, 0.0144, -1], [3, 1.4384, 1], [2, 0.3145, -1], [1, 28.5000, 1], [2, 0.0185, -1], [3, 4.0984, -1], [2, 2.8742, 1], [1, 6.5000, 1], [2, 0.1234, -1], [3, 0.2107, 1], [4, 0.0043, -1], [3, 1.4384, 1]]

# --- BALL 模型 (Class ID: 1) ---
T_BALL = 50 
ALPHA_BALL = np.array([1.9990, 0.7013, 0.6547, 0.3795, 0.5164, 0.5018, 0.4039, 0.4951, 0.4486, 0.4014, 0.3144, 0.3324, 0.3621, 0.4137, 0.3190, 0.2569, 0.2662, 0.3171, 0.3138, 0.2498, 0.3084, 0.2083, 0.2044, 0.2185, 0.2728, 0.2140, 0.1813, 0.1937, 0.2318, 0.2026, 0.1585, 0.1426, 0.2049, 0.1884, 0.1516, 0.1671, 0.1879, 0.1545, 0.1529, 0.1523, 0.1879, 0.1733, 0.1578, 0.1630, 0.1399, 0.1563, 0.1637, 0.1504, 0.1343, 0.1282], dtype=float) 
STUMPS_BALL = [[2, 0.0426, -1], [3, 1.3650, 1], [4, 0.0001, -1], [2, 0.9415, 1], [1, 33.5000, -1], [5, 2.7853, 1], [1, 13.5000, 1], [5, 0.0935, -1], [1, 33.5000, -1], [3, 1.3650, 1], [5, 0.1092, -1], [3, 0.0893, 1], [2, 0.0195, -1], [4, 0.0118, -1], [3, 1.3650, 1], [4, 0.0000, -1], [1, 6.5000, 1], [2, 0.0175, -1], [4, 0.1125, -1], [2, 0.0425, 1], [2, 0.0175, -1], [3, 2.3362, -1], [2, 2.1541, 1], [3, 0.0850, 1], [5, 0.1092, -1], [1, 8.5000, 1], [2, 0.0195, -1], [4, 0.0072, -1], [3, 1.3650, 1], [2, 1.0717, -1], [2, 1.5906, 1], [1, 4.5000, 1], [4, 0.0001, -1], [3, 1.3650, 1], [2, 1.0717, -1], [3, 0.0096, -1], [3, 0.0840, 1], [4, 0.0002, -1], [2, 0.9415, 1], [1, 24.5000, -1], [2, 0.0195, -1], [1, 4.5000, 1], [2, 0.0175, -1], [3, 2.9066, -1], [3, 0.0096, -1], [5, 0.0740, 1], [5, 0.1092, -1], [2, 0.0429, 1], [4, 0.0000, -1], [5, 1.4887, 1]]

# === 2. 門檻參數和常數 ==============================================
POST_THR = 0.60         
MIN_RANGE = 0.05
EPS = 1e-6
MAX_RANGE_M = 1.0 
LASER_FRAME_ID = "base_scan" 
BOX_THICKNESS_M = 0.10 

# --- DBF 參數 ---
MAX_DIST_FOR_ASSOCIATION = 0.30 # 追蹤關聯最大距離 (米)
TRACK_INACTIVITY_LIMIT = 0.5    # 軌跡失效時間 (秒)
MIN_PROB_TO_REPORT = 0.50       # 報告和顯示的最低存在機率

# === 3. 離散型 Bayes Filter 參數 ===================================

# 狀態: X_t 屬於 {0: 不存在, 1: 存在}
# T (Motion Model): P(X_t | X_{t-1})
T = np.array([
    [0.95, 0.10],  # P(X_t=0 | X_{t-1}=0), P(X_t=0 | X_{t-1}=1)
    [0.05, 0.90]   # P(X_t=1 | X_{t-1}=0), P(X_t=1 | X_{t-1}=1)
])

# Z (Sensor Model): P(Z_t | X_t) (Z_t=1: 偵測到, Z_t=0: 未偵測到)
P_DETECT_GIVEN_PRESENT = 0.85 # AdaBoost 準確率
P_DETECT_GIVEN_ABSENT = 0.05  # AdaBoost 誤報率

Z = np.array([
    [1 - P_DETECT_GIVEN_ABSENT, 1 - P_DETECT_GIVEN_PRESENT],  # P(Z_t=0 | X_t=0), P(Z_t=0 | X_t=1)
    [P_DETECT_GIVEN_ABSENT, P_DETECT_GIVEN_PRESENT]           # P(Z_t=1 | X_t=0), P(Z_t=1 | X_t=1)
])

# === 4. 核心 AdaBoost 與輔助函數 =====================================

def stump_predict(x5, j, theta, s):
    """對原始特徵 x_j 進行單一樹樁預測 (輸出 +1 或 -1)。"""
    feature_index = j - 1 
    x_j_raw = x5[feature_index]
    if s * (x_j_raw - theta) >= 0:
        return 1
    else:
        return -1

def adaboost_predict_proba(x5, alpha, stumps):
    """計算 AdaBoost 強分類器的分數 F(x) 並轉換為 P(Y=1|x) 。"""
    F_x = 0.0
    for t in range(len(alpha)):
        alpha_t = alpha[t]
        j, theta, s = stumps[t]
        h_t = stump_predict(x5, j, theta, s) 
        F_x += alpha_t * h_t
    F_x = np.clip(F_x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-F_x))

def svd_mean_perp_residual(x, y):
    """直線擬合的平均垂直殘差。"""
    X = np.stack([x - x.mean(), y - y.mean()], axis=1)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        perp = Vt[-1, :]
        proj = X @ perp[:, None]
        return float(np.mean(np.abs(proj)))
    except np.linalg.LinAlgError:
        return 0.0

def cluster_by_gaps(angles, ranges, angle_gap_rad, range_jump_m):
    """依角度和距離跳躍進行分段"""
    n = len(angles)
    if n == 0: return []
    clusters, start = [], 0
    for i in range(1, n):
        if (abs(angles[i]-angles[i-1]) > angle_gap_rad or
            abs(ranges[i]-ranges[i-1]) > range_jump_m or
            not np.isfinite(ranges[i])):
            clusters.append((start, i))
            start = i
    clusters.append((start, n))
    return clusters

# === 5. 離散型 Bayes Filter 類別  ============================

class DiscreteBayesFilter:
    def __init__(self, initial_x, initial_y, class_id, width, theta):
        self.x = initial_x
        self.y = initial_y
        self.class_id = class_id
        self.last_update_time = time.time()
        
        # belief: [P(X_t=0), P(X_t=1)]^T (不存在的機率, 存在的機率)
        self.belief = np.array([0.01, 0.99]).reshape((2, 1)) # 初始假設存在
        self.id = 0 
        
        self.marker_width = width
        self.marker_theta = theta

    def get_likelihood_vector(self, is_detected):
        """根據 AdaBoost 是否偵測到，選擇感測器模型中的量測似然向量 Z。"""
        global Z
        if is_detected:
            # P(Z_t=1 | X_t)
            return Z[1, :].reshape((2, 1)) 
        else:
            # P(Z_t=0 | X_t)
            return Z[0, :].reshape((2, 1)) 

    def filter_update(self, is_detected):
        """執行 Bayes Filter 的 Prediction 和 Correction 步驟。"""
        global T
        
        # 1. Prediction (Time Update): a_pred = T' * a_prev
        a_pred = T.T @ self.belief 
        
        # 2. Correction (Measurement Update)
        b = self.get_likelihood_vector(is_detected)
        
        # 3. Correction + Normalize
        a = b * a_pred
        self.belief = a / np.sum(a)
        
        self.last_update_time = time.time()
        
        # 返回 P(X_t=1 | Z_{1:t})
        return float(self.belief[1]) 

# === 6. ROS 節點類別 (追蹤管理與輸出) =================================

class UnifiedDetectorNode:
    def __init__(self):
        self.min_n = int(rospy.get_param("~min_n", 3)) 
        self.angle_gap_deg = float(rospy.get_param("~angle_gap_deg", 3.0))
        self.range_jump_m = float(rospy.get_param("~range_jump_m", 0.15))
        self.max_range_m = float(rospy.get_param("~max_range_m", MAX_RANGE_M))
        self.post_thr = float(rospy.get_param("~post_thr", POST_THR)) 
        
        self.pub = rospy.Publisher("/detected_object", Float32MultiArray, queue_size=10)
        self.marker_pub = rospy.Publisher("/detection_markers", Marker, queue_size=10)
        
        self.active_trackers = {} # {tracker_id: DiscreteBayesFilter 實例}
        self.next_tracker_id = 1
        
        rospy.Subscriber("/scan", LaserScan, self.cb_scan, queue_size=1)
        rospy.loginfo(f"UnifiedDetectorNode started with Discrete Bayes Filter (Tracking Probability)")

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [x, y, z, w]

    def create_marker(self, x, y, tracker_id, class_id, label, dist, width=0.1, theta=0.0):
        marker = Marker()
        marker.header.frame_id = LASER_FRAME_ID  
        marker.header.stamp = rospy.Time.now()
        marker.ns = "detected_objects"
        marker.id = tracker_id 
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0 
        
        if class_id == 1: # Ball (紅)
            marker.type = Marker.SPHERE
            marker.scale.x = 0.15; marker.scale.y = 0.15; marker.scale.z = 0.15
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.8  
            marker.pose.orientation.w = 1.0 
        else: # Box (藍)
            marker.type = Marker.CUBE
            marker.scale.x = width           
            marker.scale.y = BOX_THICKNESS_M 
            marker.scale.z = BOX_THICKNESS_M 
            q = self.euler_to_quaternion(0, 0, theta)
            marker.pose.orientation.x = q[0]; marker.pose.orientation.y = q[1]; marker.pose.orientation.z = q[2]; marker.pose.orientation.w = q[3]
            marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 0.8
            
        marker.lifetime = rospy.Duration(0.5) 
        return marker


    def cb_scan(self, scan: LaserScan):
        
        # --- 1. 數據預處理與 AdaBoost 偵測 ---
        n = len(scan.ranges)
        if n == 0: return
        angles = scan.angle_min + np.arange(n, dtype=float) * scan.angle_increment
        ranges = np.asarray(scan.ranges, dtype=float)
        m = min(angles.size, ranges.size)
        angles, ranges = angles[:m], ranges[:m]
        valid = np.isfinite(ranges) & (ranges > 0) & (ranges <= self.max_range_m)
        angles, ranges = angles[valid], ranges[valid]
        if angles.size < self.min_n: return

        clusters = cluster_by_gaps(angles, ranges, math.radians(self.angle_gap_deg), self.range_jump_m)
        
        current_detections = [] # 儲存 (cx, cy, class_id, post, width, theta)

        for a, b in clusters:
            ang, rg = angles[a:b], ranges[a:b]
            if len(rg) < self.min_n: continue
            x, y = rg * np.cos(ang), rg * np.sin(ang)
            npts = float(len(rg))
            sigma = float(np.std(rg))
            width = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))
            mean_perp = svd_mean_perp_residual(x, y)
            sc = float(1.0 / (1.0 + mean_perp))
            rc = float((np.max(rg) - np.min(rg)) / (np.mean(rg) + EPS))
            x5 = np.array([npts, sigma, width, sc, rc], dtype=float) 
            theta = float(math.atan2(y[-1] - y[0], x[-1] - x[0]))

            p_ball = adaboost_predict_proba(x5, ALPHA_BALL, STUMPS_BALL)
            p_box = adaboost_predict_proba(x5, ALPHA_BOX, STUMPS_BOX)

            label = None; post = 0.0; class_id = 0
            if max(p_ball, p_box) >= self.post_thr: 
                if p_box >= p_ball:
                    label, post, class_id = "box", p_box, 2
                else:
                    label, post, class_id = "ball", p_ball, 1
            else:
                continue
                
            cx = float(np.mean(x)); cy = float(np.mean(y))
            if math.hypot(cx, cy) < MIN_RANGE: continue
            
            current_detections.append((cx, cy, class_id, post, width, theta))

        # --- 2. 追蹤：數據關聯與 DBF 更新 ---
        unmatched_trackers = list(self.active_trackers.keys())
        matched_detections_indices = []

        # 2.1. 匹配並更新 (Z_t=1)
        for i, det in enumerate(current_detections):
            cx, cy, class_id, _, width, theta = det
            min_dist = float('inf')
            best_tid = -1
            
            for tid, tracker in self.active_trackers.items():
                if tracker.class_id != class_id: continue
                dist = math.hypot(tracker.x - cx, tracker.y - cy)
                
                if dist < min_dist and dist < MAX_DIST_FOR_ASSOCIATION:
                    min_dist = dist
                    best_tid = tid

            if best_tid != -1:
                tracker = self.active_trackers[best_tid]
                tracker.filter_update(is_detected=True)
                
                # 更新追蹤器的位置和屬性
                tracker.x = cx
                tracker.y = cy
                tracker.marker_width = width
                tracker.marker_theta = theta
                
                matched_detections_indices.append(i)
                unmatched_trackers.remove(best_tid)

        # 2.2. 未匹配的追蹤器：假設未被偵測到 (Z_t=0)
        for tid in unmatched_trackers:
            self.active_trackers[tid].filter_update(is_detected=False)

        # 2.3. 未匹配的偵測：初始化新的追蹤器
        for i, det in enumerate(current_detections):
            if i not in matched_detections_indices:
                cx, cy, class_id, _, width, theta = det
                
                new_tracker = DiscreteBayesFilter(cx, cy, class_id, width, theta)
                new_tracker.id = self.next_tracker_id
                self.active_trackers[self.next_tracker_id] = new_tracker
                self.next_tracker_id += 1


        # --- 3. 輸出與清理 ---
        current_time = time.time()
        trackers_to_delete = []
        
        label_map = {1: "ball", 2: "box"} 

        for tid, tracker in self.active_trackers.items():
            
            estimated_prob = float(tracker.belief[1]) # P(X_t=1)
            x_est, y_est = tracker.x, tracker.y 
            
            # 計算物體與 Sensor (原點) 的距離
            distance_to_sensor = math.hypot(x_est, y_est)
            
            # 清理低機率或失效的軌跡
            if estimated_prob < 0.1 or current_time - tracker.last_update_time > TRACK_INACTIVITY_LIMIT:
                trackers_to_delete.append(tid)
                continue
            
            # 僅輸出機率高於門檻的追蹤器
            if estimated_prob < MIN_PROB_TO_REPORT:
                continue

            # ---  Log 輸出：包含 label 字串和距離  ---
            label_str = label_map.get(tracker.class_id, "unknown")
            rospy.loginfo(
                f"ESTIMATED (DBF): ID={tid}, label={label_str}, "
                f"pos=({x_est:.2f}, {y_est:.2f}), "
                f"Dist={distance_to_sensor:.2f}m, "
                f"Prob_Exist={estimated_prob:.3f}"
            )
    
            # (object_x, object_y, probability, object_index)
            msg = Float32MultiArray()
            msg.data = [x_est, y_est, estimated_prob, float(tid)]
            self.pub.publish(msg)
            
            # 發佈 Rviz Marker 
            marker = self.create_marker(
                x_est, y_est, tid, tracker.class_id, 
                label_str, distance_to_sensor, 
                width=tracker.marker_width, theta=tracker.marker_theta
            )
            self.marker_pub.publish(marker)

        # 執行刪除 
        for tid in trackers_to_delete:
            del_marker = Marker()
            del_marker.header.frame_id = LASER_FRAME_ID
            del_marker.ns = "detected_objects"
            del_marker.id = tid
            del_marker.action = Marker.DELETE
            self.marker_pub.publish(del_marker)
            del self.active_trackers[tid]


def main():
    rospy.init_node("unified_adaboost_dbf_detector", anonymous=False)
    T_BOX_EXPECTED = 50
    T_BALL_EXPECTED = 50
    if len(ALPHA_BOX) != T_BOX_EXPECTED or len(STUMPS_BOX) != T_BOX_EXPECTED:
        rospy.logerr(f"錯誤: BOX 參數數量不一致 (預期 {T_BOX_EXPECTED})。請檢查複製是否完整。")
        return
    if len(ALPHA_BALL) != T_BALL_EXPECTED or len(STUMPS_BALL) != T_BALL_EXPECTED:
        rospy.logerr(f"錯誤: BALL 參數數量不一致 (預期 {T_BALL_EXPECTED})。請檢查複製是否完整。")
        return

    UnifiedDetectorNode()
    rospy.spin()

if __name__ == "__main__":
    main()
