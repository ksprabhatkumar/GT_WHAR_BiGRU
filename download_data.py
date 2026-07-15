import os
import urllib.request
import zipfile

# 1. Define the URLs and paths
dsads_url = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"
target_dir = os.path.join("har_data", "DSADS")
zip_path = "dsads_temp.zip"

# 2. Create the har_data/DSADS directory
os.makedirs(target_dir, exist_ok=True)

# 3. Download the file
print("Downloading DSADS dataset from UCI... (This might take a few minutes)")
urllib.request.urlretrieve(dsads_url, zip_path)
print("Download complete!")

# 4. Extract the zip file directly into har_data/DSADS
print("Extracting files...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

# 5. Cleanup
os.remove(zip_path)
print(f"✅ Success! Data is now located in: {os.path.abspath(target_dir)}")