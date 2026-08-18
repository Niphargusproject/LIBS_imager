# CAD

STEP models of the instrument, licensed under CERN-OHL-W-2.0. They open in any CAD
package that reads STEP (FreeCAD, Fusion 360, SolidWorks, Onshape).

| File | Content |
| --- | --- |
| `Screen_libs_v1_imaging_v2026.step.zip` | the laser assembly: focusing head, dichroic mount, collimators, collection arm, camera, and the plate they are mounted on |
| `Screen_libs_pulse_controller_box.step.zip` | enclosure of the electronics, with the motion controller, the pulse controller and the internal controller |
| `Screen_libs_sample_stage_plates.step` | adapter plates between the linear stages and the sample stage |
| `Screen_libs_Z_mounting_plate.step` | plate that carries the Z stage on the vertical optical breadboard |
| `End_Stop_Board_libs.step` | end-stop board of the stages |

Two of the models are zipped because they are 238 MB and 54 MB uncompressed, over
the limit GitHub accepts for a single file. Unzip them before opening; any
archiver, including Windows Explorer, will do.

Fig. 3 of the article is the optical layout of the laser assembly, Fig. 6 shows the
subsystems as built, and Fig. 8 is a render of the first model above with the
enclosure shells removed. Section 5 gives the assembly order and the fasteners. The frame
itself is standard 20 × 40 mm V-slot profile and is not modeled here; its dimensions
are in Section 5 and in the bill of materials.
