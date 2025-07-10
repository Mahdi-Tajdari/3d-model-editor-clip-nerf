# --- STEP 3.4: Write scripts/run_clip_test.py ---
%%writefile scripts/run_clip_test.py
import os
import sys

# Add the project root to the Python path
# This allows importing modules like 'src.clip_module.clip_loader'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now you can import run_clip_basic_test from main.py
from src.main import run_clip_basic_test

if __name__ == "__main__":
    print("--- Running CLIP basic test via script ---")
    run_clip_basic_test()
    print("\n--- CLIP basic test completed successfully ---")