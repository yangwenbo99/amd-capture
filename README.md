# Automatic Image Capturing Scripts

- `server/`
    - `server_main.py`: the HTTP server to be deployed on the laptop to capture images
    - `download.py`: Download from the server to the local machine
- `display/`
    - `mpv_driver.py`: Control the TV (in HDR mode) to display videos
- `calculation/`: scripts for calculation certain properties
    - `color_temp.py`: calculate the conversion multiplier between two colour temperature 
        - AI written, but is probably right. 
- `misc_scripts/`
    - `embed_color_checker.py`: overlay a color-checker onto each image in a
      directory and write results to a separate directory (uses Pillow).
      The checker is placed with its top edge around h/2 (small random jitter),
      and scaled as large as possible while constrained to <= 1/2 image width
      and <= 1/4 image height.


## HTTP Server

TL;DR: You sent an HTTP POST request to the server; the server capture the image and save it to the disk.  If you want to download the images, the server also supports it. 

**Warning**: only deploy this script on a trusted network.  From a quick check, I can conclude it is full of security holes.

### API Endpoints

* `POST /capture` → trigger capture and wait for result
* `GET /health` → returns `{"ok": true}`
* `GET /last-status` → returns the last run result
* `GET /files` to list all files in the monitored directory
    - Arguments:
        - `pattern` (optional): a glob pattern to filter files, e.g., `*.jpg`
        - `modified_after` (optional): a timestamp to filter files modified after that time, e.g., `2024-06-01T00:00:00`
        - `modified_before` (optional): a timestamp to filter files modified before that time, e.g., `2024-06-30T23:59:59`
* `DELETE /files/{name}` to list all files in the monitored directory
* `GET /pull` to download a file from that directory

### Deploy Instructions

Install Mambaforge, then in the forge prompt, run:

```bash
mamba create -n capture python=3.11 -y
mamba activate capture
pip install fastapi uvicorn pywinauto pydantic
```

If prompted to init the shell, follow the instruction and start over. 

## Download Script

Example:

```bash
python download_all_files.py \
  --server http://192.168.1.10:8765 \
  --output-dir downloads \
  --pattern "*.png" \
  --modified-after "2026-04-08T12:30:00" \
  --modified-before "2026-04-08T13:00:00"
```

## TODOs

- Hardware setup instructions

## misc_scripts

### Embed a color checker into each image

Install dependency:

```bash
python3 -m pip install pillow
```

Run:

```bash
python3 misc_scripts/embed_color_checker.py \
  --input-dir in_images \
  --output-dir out_images \
  --checker color_checker.png
```
