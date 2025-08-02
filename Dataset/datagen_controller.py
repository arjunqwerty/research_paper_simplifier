'''import subprocess

# Initialize start index
start_index = int(input("Enter the initial start index: "))
increment_value = 60  # Change if needed

while True:
    print(f"\n==== Running datagen_1_download.py with start index {start_index} ====\n")
    download_process = subprocess.Popen(["python", "datagen_1_download.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    download_process.stdin.write(f"{start_index}\n")
    download_process.stdin.flush()
    
    for line in download_process.stdout:
        print(line.strip())  # Print script output in real-time

    download_process.wait()  # Wait for process to complete

    print("\n==== Running datagen_2_summarize.py ====\n")
    summarize_process = subprocess.Popen(["python", "datagen_2_summarize.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    for line in summarize_process.stdout:
        print(line.strip())  
        if "ALL TOKENS EXHAUSTED" in line:
            print("Stopping execution: Tokens exhausted in datagen_2_summarize.py")
            exit(0)  # Exit the controller script

    summarize_process.wait()

    print("\n==== Running datagen_3_storygen.py ====\n")
    story_process = subprocess.Popen(["python", "datagen_3_storygen.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in story_process.stdout:
        print(line.strip())  
        if "ALL TOKENS EXHAUSTED" in line:
            print("Stopping execution: Tokens exhausted in datagen_3_storygen.py")
            exit(0)

    story_process.wait()

    # Increment the start index for the next loop
    start_index += increment_value
    print(f"\n==== Incrementing start index to {start_index} ====\n")
'''

'''import subprocess
import sys

# Initialize start index
start_index = int(input("Enter the initial start index: "))
increment_value = 60  # Change if needed

def run_script(script_name, input_text=None):
    """Runs a script, prints output in real-time, and stops if 'ALL TOKENS EXHAUSTED' is found."""
    process = subprocess.Popen(["python", script_name], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT,  # Combine stderr with stdout
                               bufsize=1, 
                               universal_newlines=True)  

    if input_text:
        process.stdin.write(f"{input_text}\n")
        process.stdin.flush()
    
    for line in process.stdout:
        sys.stdout.write(line)  # Print in real-time
        sys.stdout.flush()
        
        if "ALL TOKENS EXHAUSTED" in line:
            print(f"\nStopping execution: Tokens exhausted in {script_name}")
            process.kill()
            sys.exit(0)

    process.wait()

while True:
    print(f"\n==== Running datagen_1_download.py with start index {start_index} ====\n")
    run_script("datagen_1_download.py", str(start_index))

    print("\n==== Running datagen_2_summarize.py ====\n")
    run_script("datagen_2_summarize.py")

    print("\n==== Running datagen_3_storygen.py ====\n")
    run_script("datagen_3_storygen.py")

    # Increment the start index for the next loop
    start_index += increment_value
    print(f"\n==== Incrementing start index to {start_index} ====\n")
'''

import subprocess

# Initialize start index
start_index = int(input("Enter the initial start index: "))
increment_value = 100  # Change if needed

def run_script(script_name, input_text=None):
    """Runs a script and stops execution if 'ALL TOKENS EXHAUSTED' is detected."""
    
    cmd = ["python", script_name]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
    
    # If the script requires an input, send it
    if input_text:
        process.communicate(input_text + "\n")
    else:
        process.wait()  # Wait for script to complete normally

while True:
    print("\n==== Running datagen_2_summarize.py ====\n")
    run_script("datagen_2_summarize.py")

    print("\n==== Running datagen_3_storygen.py ====\n")
    run_script("datagen_3_storygen.py")

    # Increment the start index for the next loop
    start_index += increment_value
    print(f"\n==== Incrementing start index to {start_index} ====\n")

    print(f"\n==== Running datagen_1_download.py with start index {start_index} ====\n")
    run_script("datagen_1_download.py", str(start_index))
