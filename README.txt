Setup and execution:
1. pip install -r requirements.txt
2. Setup environment variable to indicate time interval and select the data 
    INTERVAL="D" for daily or INTERVAL="H" for hourly data
3. To modify list of models to train edit MODEL_CONFIGS in model_configs.py
4. python run.py



Reviewing results after execution run.py
- general result with comparision and etc. should be in ./results/YOUR_INTERVAL/general/
- saved predictions results and plots for specific model are in  ./results/YOUR_INTERVAL/YOUR_MODEL/
- for zooming plots of predictions use open_single() function from view_results.py  

Downloading NASDAQ s3 parquets files and processing them.
1. Set up AWS credentials in your system with Access key and id
2. Run "s3_download.ps1 TICKER", where ticker is i.e. AAPL or GOOGL
3. After that you can create csv files with merged data running "python merge.py"
4. python datasets.py
5. Inspect the data with visualization.ipybn notebook.
For daily csv data from yahoo go to step 4., make sure that the csv files are located in ./data/daily/

