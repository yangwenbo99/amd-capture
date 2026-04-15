Below shows an example 

Server:

Turn on all apps and don't forget to turn on auto exposure

```
    
      conda activate capture
      cd document/ccapture
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
