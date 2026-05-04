#!/bin/bash
# Re-extract source videos at higher fps to add more training frames.
# Adjacent frames at fps=12 are 80ms apart. Some will dHash-cluster
# with existing fps=4/5 frames; those that don't will add cluster
# diversity. Runs on the box only.
set -e
cd /opt/rfcai/training
sudo -u rfcai bash -c "cd /opt/rfcai/training && .venv/bin/python -c \"
import sys
sys.path.insert(0, 'scripts')
from pathlib import Path
import auto_label_videos as al
for video, family in [
    ('data/videos/2_4mm.MOV',  '2.4mm'),
    ('data/videos/2_92mm.MOV', '2.92mm'),
    ('data/videos/3_5mm.MOV',  '3.5mm'),
]:
    print(f'extracting {video} @ fps=12...')
    counts = al.process_video(Path(video), family, fps=12.0, max_crops_per_frame=3)
    print(f'  {family}: {counts}')
print('done')
\""
