import subprocess
import time
import os 

if __name__ == "__main__":
    
    t0 = time.time()
    process_path = os.path.join(os.environ['HOME'], "guanaco", "proc_generate_philosophy_debate.py")
    processes = []
    
    for i in range(1,4):
        cfg_path = os.path.join(os.environ['HOME'], "guanaco", f"FredRalph_p{i}.toml")
        process =  subprocess.Popen(['python3',process_path, cfg_path])
        processes.append(process)

    for process in processes:
        process.wait()
        
    print('All procs executed, total time elapsed: %.2f'%(time.time() - t0))