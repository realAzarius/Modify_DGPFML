import os
import shutil
import time

# read files
# create directories based on modification date
# move file to the floder
current_dir = './'
files = os.listdir(current_dir)

for f in files:
    if os.path.isdir(f) or f.endswith('.py'):
        continue

    file_path = os.path.join(current_dir, f)
    stats = os.stat(file_path)

    modified_time = stats.st_mtime
    time_local = time.localtime(modified_time)
    dt = time.strftime("%Y-%m-%d",time_local)
    new_floder = os.path.join(current_dir, dt)

    if not os.path.exists(new_floder):
        os.makedirs(new_floder)

    dst_file = os.path.join(new_floder, f)
    if os.path.exists(dst_file):
        # in case of the src and dst are the same file
        if os.path.samefile(file_path, dst_file):
            continue
        os.remove(dst_file)
        
    shutil.move(file_path, new_floder)