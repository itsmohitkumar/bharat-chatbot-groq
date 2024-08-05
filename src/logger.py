import logging
import os
from datetime import datetime

# Define log file name using current date and time
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Define the path for the log directory
log_path = os.path.join(os.getcwd(), "logs")

# Create log directory if it does not exist
os.makedirs(log_path, exist_ok=True)

# Define the full path for the log file
LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILEPATH,
    filemode='a',  # Use 'a' to append to the log file
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# Add a handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Define the logger object
logger = logging.getLogger(__name__)