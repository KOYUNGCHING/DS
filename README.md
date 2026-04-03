# DS
## Midterm_時時物體偵測+rviz
執行sop:

機器人：laser_dump.py蒐集資料

分別用 label_ball_timeslices.m ＆ label_box_timeslices.m 逐秒標記

把標記完的 csv 丟到 label_change

再把 labeled_features_multiclass_box.csv ＆ labeled_features_multiclass_ball.csv 丟到這個檔案訓練

再把訓練完的數據套到 ros_predictor.py 給機器人
