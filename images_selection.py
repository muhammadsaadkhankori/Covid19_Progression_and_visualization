import pandas as pd
import shutil
import os

metadata = r"E:\Study materials\Torch\metadata.csv"
imageDir = r"E:\Study materials\Torch\Images"
outputDir = r"E:\Study materials\Torch\Normal"

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
	if row["finding"] == 'No Finding' and row["modality"]=="X-ray":
		filename = row["filename"].split(os.path.sep)[-1]
		filePath = os.path.sep.join([imageDir, filename])
		shutil.copy2(filePath, outputDir)
