import librosa

import torch
from datasets import load_dataset
ds = load_dataset('google/MusicCaps', split='train')
import subprocess
import os
from pathlib import Path

def download_clip(
    video_id,
    output_file,
    start_time,
    end_time,
    tmpdir='/tmp/musiccaps',
    num_attempts=5,
    url_base='https://youtube.com/watch?v='
):
    status = False

    command = f"""yt-dlp --no-warnings -x --audio-format wav -f bestaudio -o "{output_file}" --download-sections "*{start_time}-{end_time}" {url_base}{video_id}""".strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        # It's a try-except-else block. If there's no exception
        # thrown then the else block is executed, i.e. if the video
        # is successfully downloaded
        else:
            break

    status = os.path.exists(output_file)
    return status, 'Downloaded'

def process(example):
    output_file = str(data_dir / f"{example['ytid']}.wav")
    print(output_file)
    status = True
    if not os.path.exists(output_file):
        status = False
        status, log = download_clip(
            video_id=example['ytid'],
            output_file=output_file,
            start_time=example['start_s'],
            end_time=example['end_s']
        )

    example['audio'] = output_file
    example['downloaded_status'] = status
    return example

from datasets import Audio
samples_to_load = 35
cores = 4
sampling_rate = 7700
writer_batch_size = 1000
data_dir = "./music_data"

ds = ds.select(range(samples_to_load))

data_dir = Path(data_dir)
# data_dir.mkdir(exist_ok=True, parents=True)

ds = ds.map(
    process,
    num_proc=cores,
    writer_batch_size=writer_batch_size,
    keep_in_memory=False
).cast_column("audio", Audio(sampling_rate=sampling_rate))

# ds = ds.map(numpy_to_tensor, num_proc=cores, writer_batch_size=writer_batch_size, keep_in_memory=False)
ds.set_format("torch", columns=["caption", "audio"])

# array, length = librosa.load("/Users/berkayg/Codes/music-project/muscall/music_data/-0Gj8-vB1q4.wav")
pass