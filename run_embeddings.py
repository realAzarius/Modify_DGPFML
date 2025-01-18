import subprocess

proc = subprocess.Popen('python ./visual_embeddings.py', shell=True)
monitor_pid = proc.pid
proc.wait()