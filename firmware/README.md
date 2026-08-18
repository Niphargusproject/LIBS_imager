# Firmware

Two Arduino sketches and the settings of the motion controller. Licensed under
GNU GPL v3. Wiring is given in Fig. 7 of the article.

## `pulse_counter_INT0/pulse_counter_INT0.ino` — pulse controller

Runs on an Arduino Uno (ATmega328P). This is the board that makes the acquisition
hardware-triggered.

* The X-axis step line of the motion controller enters on **D2 (INT0)**. Every rising
  edge increments a counter in the interrupt routine.
* When the counter reaches `step_target`, the sketch raises **D4** and the main loop
  drops it 100 µs later, then the counter restarts. The interrupt routine itself never
  waits, so no step pulse can be missed.
* D4 drives the external trigger input of the laser. The shot pitch is therefore
  `step_target / steps_per_mm` millimetres, i.e. `step_target` microsteps of stage
  travel, and does not depend on any software timing.
* Serial interface, 115200 baud, newline-terminated (uses the
  [SerialCommand](http://github.com/p-v-o-s/Arduino-SerialCommand) library):

  | Command | Effect |
  | --- | --- |
  | `STEP <n>` | set the number of microsteps between triggers |
  | `START` | attach the interrupt, i.e. arm the trigger chain |
  | `STOP` | detach the interrupt, i.e. disarm the trigger chain |
  | `COUNT` | number of triggers emitted since the last reset |
  | `RESETCOUNT` | reset that counter |
  | `H` | list the commands |
  | `?` | identify the device (`LIBS motor pulse counter v1.1`) |

  The acquisition application compares `COUNT` with the number of shots it expects
  after every line, which is how a missed or extra trigger is detected.
* `PUMP_ON`, `PUMP_OFF` and `FILTER` are accepted by this version of the sketch but do
  nothing yet; the air pump and the extraction unit are switched by hand. The commands
  are kept so that a relay board can be added on a free output without changing the
  acquisition software.

## `Libs_leds_distance_sensor_controller/Libs_leds_distance_sensor_controller.ino` — internal controller

Runs on an Arduino Nano inside the laser assembly. It switches the two illumination
LEDs and, optionally, reads a distance sensor.

* **D3** drives the micro-view LED and **A3** the macro-view LED, each through a 120 Ω
  series resistor.
* **A2** is the XSHUT (standby) line of a VL53L4CD time-of-flight distance sensor, which
  is read over I²C on SDA/SCL. The sensor is wired but not yet used for automatic focus
  following (article Section 7.8); it starts in standby.
* Serial interface, 115200 baud, newline-terminated: `?`, `H`, `ON` / `OFF` (micro LED),
  `MACRO_LED_ON` / `MACRO_LED_OFF`, `SENSOR_ON` / `SENSOR_OFF`, `DIST` (distance in mm),
  `DIST_DIAG` (distance with status and signal level).
* Needs the `STM32duino VL53L4CD` library (`vl53l4cd_class.h`) from the Arduino library
  manager.

## `Grbl_ESP32_configuration.txt` — motion controller settings

The `$`-settings of the Makerbase MKS DLC32 board running Grbl_ESP32. Restore them by
sending the lines to the controller over the serial console, then check with `$$`.
The header of the file lists the values that matter for imaging (resolution, feed
rates, accelerations) and explains what to change when the mechanics differ.

## Flashing

Both sketches are flashed with the Arduino IDE (board: *Arduino Uno* and
*Arduino Nano*, ATmega328P). No bootloader change or fuse setting is needed. Flash the
boards before they are wired into the machine, and keep the trigger cable
disconnected from the laser while testing.
