from datetime import datetime
import os

# 現在の日付と時間を取得
current_datetime = datetime.now()

# ファイルに追記
with open('pkl_update_log.txt', 'a') as f:
    f.write(str(current_datetime) + '\n')
