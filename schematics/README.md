# Schematics

`wiring_trigger_chain.png` is Fig. 7 of the article, at publication resolution. It has
two panels:

* **(a) the trigger chain.** From the X step output of the MKS DLC32 motion
  controller to the Arduino Uno pulse controller, and from there to the external
  trigger input of the laser, which fans the sync pulse out to the six spectrometers.
  This is the path that locks shot spacing to stage position.
* **(b) the internal controller.** The Arduino Nano inside the laser assembly, its two
  illumination LEDs with their 120 Ω series resistors, and the I²C connection left for
  the time-of-flight distance sensor.

`../firmware/README.md` lists the pins and the serial commands of both boards, and
Section 5 of the article walks through the wiring step by step.

The drawing is a signal-level schematic, not a PCB layout: apart from the end-stop
board in `../cad/`, the electronics are off-the-shelf boards joined by jumper wires
and screw terminals, so there is no board file to publish.
