import threading
import subprocess
import os
import time

# Shared variable
a = 0

# Mutex for thread-safe access to 'a'
mutex = threading.Lock()

def attempt_SNIF():
    start_snif = time.time()
    global a
    # Acquire the lock to ensure exclusive access to 'a'
    with mutex:
        # Store the current value of 'a' in a temporary variable and increment 'a'
        a_temp = a
        a += 1

    # Check if 'a' is less than 6975
    if a_temp < 2686: # / put +1 du cas max quue vous avez 
        # Perform the 'snif' action (represented here by a print statement)
        result = subprocess.run(["python3", "SNIF_sim_" + format(a, '04d') + ".py"], capture_output=True, text=True)  # on recupère les sorties terminal de SNIF pour les mettre dans un fichier d'execution et faciliter le debug
        end_snif = time.time()
        print(f"Performing SNIF for case {a_temp} in {format(end_snif-start_snif, '.2f')} sec")
        print(result.stdout)
        # Return True to indicate that the process is not finished
        return True
    else:
        # Return False to indicate that the process is finished
        return False

def main_thread(thread_id):
    while attempt_SNIF():
        # Print the current value of 'a' and the thread ID
        print(f"a is {a}, thread {thread_id} is talking")

# List to hold the thread objects
threads = []
time_start = time.time()
# Create and start 63 threads
for i in range(63): #mettez ici le nombre de coeurs (cores) de votre machine ('$nproc --all' sur linux)
    thread = threading.Thread(target=main_thread, args=(i,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

	
time_end = time.time()
print(f"Process end in {format(time_end-time_start, '.2f')} sec")
