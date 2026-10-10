#!/bin/bash
# Pi 5 setup: ReSpeaker + offline speech. Idempotent.
set -e
sudo apt update && sudo apt install -y python3-venv portaudio19-dev libsndfile1 git espeak-ng
python3 -m venv .venv-v2 && . .venv-v2/bin/activate
pip install -r requirements.txt
# ReSpeaker seeed-voicecard (Pi 5 needs patched DT overlay; reboot after)
if [ ! -d /opt/seeed-voicecard ]; then git clone https://github.com/respeaker/seeed-voicecard.git /tmp/seeed-voicecard; fi
echo "--- record pins: arecord -l should show seeed-2mic-voicecard ---"
echo "--- pre-cache models while online, then run HF_HUB_OFFLINE=1 python app/serve.py ---"
