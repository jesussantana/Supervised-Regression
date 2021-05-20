import sys
new_path = '../scripts/'
if new_path not in sys.path:
    sys.path.append(new_path)

import os
os.environ["SETTINGS_MODULE"] = 'settings'

import settings

# Path folder configuration
# ===============================================================================

path = '../data/'
file = 'raw/DelayedFlights.csv'

df_raw = pd.read_csv(path+file)