Below shows an example 

Server:

Turn on all apps and don't forget to turn on auto exposure

In miniforge prompt, run:

```
    
      conda activate capture
      cd Documents/capture
      python main.py
```


Display
```
python mpv_driver.py --media-root /home/yang/Downloads/wm-people/checkers/checker-res/
```


Control
```
python3 control/control_capture_session.py \
  --display http://localhost:8080 \
  --capture http://192.168.0.101:48765 \
  --k-captures 3 \
  --brightness-scale-min 0.70 \
  --brightness-scale-max 1.30 \
  --kelvin-min 4500 \
  --kelvin-max 7500 \
  --image-list /home/yang/Downloads/wm-people/flist/flist-jpgs.txt \
  --delay-before-capture 3.0 \
  --download-dir captures-download \
  --retry-delay-sec 4 \
  --step-log captures/steps.tsv
```

```
python3 control/control_capture_session.py \
  --display http://localhost:8080 \
  --capture http://192.168.0.101:48765 \
  --k-captures 3 \
  --brightness-scale-min 0.70 \
  --brightness-scale-max 1.30 \
  --kelvin-min 4500 \
  --kelvin-max 7500 \
  --image-list /home/yang/Downloads/wm-people/flist/flist-jpgs.txt \
  --delay-before-capture 3.0 \
  --download-dir captures-download-checkers \
  --retry-delay-sec 4 \
  --step-log captures-checkers/steps.tsv
```


### For testing projecting cameras' settings

```
python mpv_driver.py \
    --media-root ../../camera-projection \
    --mpv-gpu-api vulkan 

python3 mpv_driver.py \
    --media-root ../../camera-projection \
  --bind 0.0.0.0 --port 8080 \
  --windowed \
  --mpv-vo gpu \
  --mpv-gpu-api opengl \
  --mpv-msg-level all=debug \
  --mpv-log-file /tmp/mpv-debug.log

```

```
python3 control/control_capture_session.py \
  --display http://localhost:8080 \
  --capture http://192.168.0.101:48765 \
  --k-captures 3 \
  --brightness-scale-min 0.70 \
  --brightness-scale-max 1.20 \
  --kelvin-min 3500 \
  --kelvin-max 7500 \
  --image-list /home/yang/Documents/camera-projection/flist.txt \
  --delay-before-capture 10.0 \
  --download-dir captures-download-cam \
  --retry-delay-sec 10 \
  --step-log captures-cam/steps.tsv


python3 control/control_capture_session.py \
  --display http://localhost:8080 \
  --capture http://192.168.0.101:48765 \
  --k-captures 3 \
  --brightness-scale-min 0.70 \
  --brightness-scale-max 1.20 \
  --kelvin-min 3500 \
  --kelvin-max 7500 \
  --image-list /home/yang/Documents/camera-projection/flist.txt \
  --delay-before-capture 10.0 \
  --download-dir captures-download-cam-2 \
  --retry-delay-sec 10 \
  --step-log captures-cam-2/steps.tsv

python3 control/control_capture_session.py \
  --display http://localhost:8080 \
  --capture http://192.168.0.101:48765 \
  --k-captures 3 \
  --brightness-scale-min 0.70 \
  --brightness-scale-max 1.20 \
  --kelvin-min 3500 \
  --kelvin-max 7500 \
  --image-list /home/yang/Documents/camera-projection/flist.txt \
  --delay-before-capture 10.0 \
  --download-dir captures-download-cam-3 \
  --retry-delay-sec 10 \
  --step-log captures-cam-3/steps.tsv \
  --augmentation-mode scripted_hdr --crop-enabled --crop-ratio 16:9
```

#### Details

Display:

```
python3 display/mpv_driver.py \
    --media-root ../camera-projection \
  --bind 0.0.0.0 --port 8080 \
  --mpv-vo gpu \
  --mpv-gpu-api opengl \
  ```



TV settings:

`cam`: original

`cam-2` and `cam-3`: adjusted


## First patch capture

An FTP server is deployed on the capturing machine (`C:\Users\Administrator\Downloads\ftpServer`).  User name: `ivc`, password: `aaa123`.
 
```
python3 display/mpv_driver.py \
    --media-root ~/Downloads/WQI-people-4k-people-llama-llama-sim/flist \
  --bind 0.0.0.0 --port 8080 \
  --mpv-vo gpu \
  --mpv-gpu-api opengl \
  ```

```bash
python3 control/control_capture_session.py \
  --display http://localhost:8080 \
  --capture http://192.168.0.101:48765 \
  --k-captures 3 \
  --brightness-scale-min 0.70 \
  --brightness-scale-max 1.20 \
  --kelvin-min 3500 \
  --kelvin-max 7500 \
  --image-list ~/Downloads/WQI-people-4k-people-llama-llama-sim/flist1k \
  --delay-before-capture 10.0 \
  --download-dir captures-download-wm-batch-1 \
  --retry-delay-sec 10 \
  --step-log captures-wm-batch-1/steps.tsv \
  --augmentation-mode scripted_hdr \
  --crop-enabled \
  --crop-ratio 16:9 \
    --crop-mode reflect_pad \
```


```bash
python3 control/control_capture_session.py \
  --display http://localhost:8080 \
  --capture http://192.168.0.101:48765 \
  --k-captures 3 \
  --brightness-scale-min 0.70 \
  --brightness-scale-max 1.20 \
  --kelvin-min 3500 \
  --kelvin-max 7500 \
  --image-list ~/Documents/camera-projection/flist.txt \
  --delay-before-capture 10.0 \
  --download-dir re-captures-202606 \
  --retry-delay-sec 10 \
  --step-log re-captures-202606/steps.tsv \
  --augmentation-mode scripted_hdr \
  --crop-enabled \
  --crop-ratio 16:9 \
    --crop-mode reflect_pad \
```

To turn off the TV:
```bash
adb connect 192.168.0.227:5555
adb shell input keyevent 26       # Pressing the power button

```
