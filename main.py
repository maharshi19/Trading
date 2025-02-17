from multiprocessing import Process
import os

# Function to run the data-fetching script
def run_data_fetcher():
    os.system("python fetch_crypto.py")

# Function to run the dashboard script
def run_dashboard():
    os.system("python dashboard.py")

if __name__ == "__main__":
    # Create separate processes for each script
    data_fetcher_process = Process(target=run_data_fetcher)
    dashboard_process = Process(target=run_dashboard)

    # Start both processes
    data_fetcher_process.start()
    dashboard_process.start()

    # Wait for both processes to complete (they'll run indefinitely in this case)
    data_fetcher_process.join()
    dashboard_process.join()
