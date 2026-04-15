# control

`control_capture_session.py` orchestrates:

- an mpv display server (from `display/mpv_driver.py`)
- a Windows capture server (from `server/server_main.py`)

It loads one image at a time on the display side, waits configurable delays
(especially **"show → capture"**), triggers capture, and optionally downloads
every captured file to a local directory.

## Example

Load two images, wait 1.2s between showing and capture, and download each
captured file:

```bash
python3 control/control_capture_session.py \
  --display http://192.168.1.2:8080 \
  --capture http://192.168.1.10:48765 \
  --image "set1/img001.exr" \
  --image "set1/img002.exr" \
  --delay-before-capture 1.2 \
  --download-dir captures
```

Sweep settings \(k captures per image\) over brightness and Kelvin ranges:

```bash
python3 control/control_capture_session.py \
  --display http://192.168.1.2:8080 \
  --capture http://192.168.1.10:48765 \
  --image-dir set1 \
  --k-captures 5 \
  --brightness-scale-min 0.95 \
  --brightness-scale-max 1.05 \
  --kelvin-min 5500 \
  --kelvin-max 7500 \
  --delay-before-capture 1.0 \
  --download-dir captures
```

Use a list file:

```bash
python3 control/control_capture_session.py \
  --display http://192.168.1.2:8080 \
  --capture http://192.168.1.10:48765 \
  --image-list images.txt \
  --delay-before-capture 0.75 \
  --download-dir captures
```

## Delay knobs

- `--delay-before-load`: wait before loading each image
- `--delay-after-load`: wait after `/load` returns
- `--delay-after-simulate`: wait after `/simulate` returns
- `--delay-before-capture`: wait between showing and capture trigger
- `--delay-after-capture`: wait after capture completes

## Retries and step timestamps

- `--capture-retries`: retry capture on timeouts
- `--retry-delay-sec`: sleep between retries
- `--step-log`: write a TSV log with per-step start/end timestamps and duration
  (defaults to `steps.tsv` inside `--download-dir` when downloading)

## Downloading

If `--download-dir` is provided, the script downloads every path returned in
the capture server's `files_found` using its `/pull` endpoint.

Downloaded files are saved using **only the file stem** (no extension). If
multiple captures would map to the same stem, you must use `--overwrite` or
they will be skipped.
