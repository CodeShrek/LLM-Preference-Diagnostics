import json
import subprocess
import os

def main():
    # 1. Load original stimuli
    with open('experiments/m3v2/m3v2_stimuli.json', 'r') as f:
        stim = json.load(f)
        
    # 2. Modify quality_threshold to 10 for all primary dilemmas
    for d in stim['primary_domain']['dilemmas']:
        d['quality_threshold'] = 10
        
    # 3. Save to a temporary file in the followup dir
    stim_path = 'experiments/followups/threshold_relevance/m3v2_stimuli_thresh10.json'
    with open(stim_path, 'w') as f:
        json.dump(stim, f, indent=2)
        
    print(f"Created overridden stimuli at {stim_path}")
    
    # 4. Run the frozen m3v2_runner.py but point to the new stimuli
    out_csv = 'results/followups/threshold_relevance_main.csv'
    os.makedirs('results/followups', exist_ok=True)
    
    cmd = [
        'python', 'experiments/m3v2/m3v2_runner.py',
        '--condition', 'main',
        '--output', out_csv,
        '--stimuli', stim_path,
        '--sleep-seconds', '0.0'
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("Done! threshold_relevance experiment complete.")

if __name__ == "__main__":
    main()
