#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laser_dump.py
----------------------------------------
用途：
  - 從 /scan 訂閱雷射掃描資料
  - 每幀記錄 (t, angle, range)
  - 可設定掃描時間 _duration 與輸出檔案 _save_path
  - 時間到自動停止並存檔
"""

import rospy
from sensor_msgs.msg import LaserScan
import time

class LaserDumper:
    def __init__(self):
        rospy.init_node("laser_dump", anonymous=True)

        # === 參數 ===
        self.topic     = rospy.get_param("~scan_topic", "/scan")
        self.duration  = float(rospy.get_param("~duration", 60.0))   # 預設 60 秒
        self.save_path = rospy.get_param("~save_path", "/home/mrlrobot/Desktop/laser_dump.dat")

        self.start_time = None
        self.file = open(self.save_path, "w")
        self.file.write("# t angle range\n")

        # 訂閱雷射資料
        rospy.Subscriber(self.topic, LaserScan, self.callback)
        rospy.loginfo(f"[laser_dump] Listening to {self.topic} for {self.duration:.1f}s → {self.save_path}")

    def callback(self, msg: LaserScan):
        # 初始化起始時間
        if self.start_time is None:
            self.start_time = time.time()

        # 計算當前時間差
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            rospy.loginfo(f"[laser_dump] Done ({elapsed:.1f}s). File saved to: {self.save_path}")
            self.file.close()
            rospy.signal_shutdown("Recording finished")
            return

        # 寫入每個掃描角度
        angle = msg.angle_min
        for r in msg.ranges:
            if r > 0.0 and r < msg.range_max:
                self.file.write(f"{elapsed:.6f} {angle:.6f} {r:.6f}\n")
            angle += msg.angle_increment

if __name__ == "__main__":
    try:
        node = LaserDumper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
