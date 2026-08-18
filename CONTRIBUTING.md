# Contributing

Questions, bug reports and improvements are welcome, whether they concern the
mechanics, the electronics or the code. Open an issue on
<https://github.com/Niphargusproject/LIBS_imager/issues>, or write to
cburlet@naturalsciences.be.

Useful things to include in an issue:

* what you built or ran, and on which operating system and Python version;
* for acquisition problems, the log written next to the cube and the output of
  `test_full_hardware.py`;
* for a missed or extra trigger, the pulse count reported by the controller
  (`COUNT`) against the number of shots the application expected;
* for a mechanical or optical change, a photograph or a sketch. A STEP file is even
  better.

## Pull requests

Keep changes focused, and say in the description what you tested on real hardware,
since most of this code cannot be exercised without a laser and spectrometers.

Please respect the licensing of the repository, which is per file and declared in
`REUSE.toml`:

* new CAD in `cad/` is `CERN-OHL-W-2.0`;
* new firmware and application code is `GPL-3.0-only`;
* new documentation is `CC-BY-4.0`.

If you add files of another kind, or third-party material, add an `[[annotations]]`
entry for them in `REUSE.toml` and check it with `pipx run reuse lint`. Do not commit
the Avantes DLL, the Avantes manual or the Basler runtime: they are proprietary and
`LICENSING.md` explains how users obtain them.

## Hardware safety

The instrument runs a Class 4 pulsed laser. Please do not propose changes that
remove or bypass the interlock, the enclosure or the trigger-arming logic of the
pulse controller.
