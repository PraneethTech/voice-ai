# Hardware wiring — tobi v2

Pi 5 GPIO is 3.3V logic. Use gpiozero (not old RPi.GPIO). Reference: https://gpiozero.readthedocs.io/en/latest/recipes.html
Pinout: https://pinout.xyz

## 1. ReSpeaker Lite → Pi 5
USB-C of the XMOS XU316 → any Pi USB port. It enumerates as a USB sound card (mic + 5W amp
speaker out). No driver. Speaker (4-8Ω, 3-5W) → JST-PH2.0 port; 3.5mm jack overrides the JST.
Wiki: https://wiki.seeedstudio.com/respeaker_lite_pi5/ · FAQ: https://wiki.seeedstudio.com/respeaker_lite_faq/

## 2. MTS-202 DPDT mute switch (physical mute)
One pole drives GPIO, the other stays spare (true analog break later if wanted):
- Pole A common → GPIO17 (input, internal pull-up)
- Pole A position 1 → GND, position 2 → 3.3V
- Read with `gpiozero.Button(17, bounce_time=0.05)`; poll in a thread or edge callback.
Muted state → red LED; live → green LED (below). The pipeline gates capture on this flag.

## 3. Status LEDs (MIC LIVE / MUTED)
- Green LED anode → GPIO27 via 330Ω → cathode → GND
- Red LED anode → GPIO22 via 330Ω → cathode → GND
States: live=green, muted=red, thinking=both blink slow, meeting=add purple (or WS2812 on GPIO18
with a level shifter; `rpi_ws281x`). Video: McWhorter switch+LED lesson yL5BNA_Ex6s.

## 4. Meeting-mode push button
Momentary button between GPIO23 and GND, `Button(23, pull_up=True, bounce_time=0.05)` —
toggle meeting mode (purple). Same pattern on the UI button tonight.

## 5. KY-040 rotary encoder (volume)
- CLK → GPIO19, DT → GPIO26, SW → GPIO13, + → 3.3V, GND → GND
- gpiozero RotaryEncoder or polling; map ±1 detent to ±5% volume via `alsamixer`/`pactl`.
Video (full Pi, not Pico): 4kypUKRMGYk.

## 6. 3.4" SPI TFT (ILI9486 class)
Note: Waveshare has no 3.4" SPI panel — generic "3.4 inch SPI" boards are ILI9486 clones; the
Waveshare 3.5" RPi LCD (A) guides apply. Wiring: VCC 3.3V, GND, DIN GPIO10 (MOSI), CLK GPIO11
(SCLK), CS GPIO8 (CE0), DC GPIO25, RST GPIO17→ use GPIO27 if 17 is taken by the mute switch —
keep one pin per function, record yours here. Driver: lcdwiki / Waveshare script.
Video: vCAGzLGTUk4. Wiki: https://www.waveshare.com/wiki/3.5inch_RPi_LCD_(A)

## 7. Power
Official 27W USB-C for the Pi. The ReSpeaker Lite takes 5V from USB (or its 5V pad). Speaker
load is small (≤5W music peak). Don't power the Pi from a hub while the cooler runs.

## Keep-outs
- Keep the speaker + its wires away from the mic array (barge-in echo).
- The TFT shares SPI pins with nothing else here; HATs would conflict with the ReSpeaker USB path.
