////////////////////////////////////////////////////////////////////////////////
// LIBS Internal Controller                                                   //
// Geological Survey of Belgium - 02/2026                                     //
//                                                                            //
// Serial command interface for controlling LEDs and a VL53L4CD ToF distance  //
// sensor used in a LIBS (Laser-Induced Breakdown Spectroscopy) setup.        //
//                                                                            //
// Features:                                                                  //
//   - Micro and macro LED control (on/off)                                   //
//   - VL53L4CD time-of-flight distance sensor with on-demand measurements    //
//   - Sensor power management via XSHUT (hardware standby)                   //
//                                                                            //
// Serial protocol: 115200 baud, newline-terminated string commands.          //
// Send "H" for a list of available commands.                                 //
//                                                                            //
// Pin assignments:                                                           //
//   LED_BUILTIN  - status LED                                                //
//   D3           - micro LED                                                 //
//   A3           - macro LED                                                 //
//   A2           - VL53L4CD XSHUT                                            //
//   SDA/SCL      - VL53L4CD I2C                                              //
////////////////////////////////////////////////////////////////////////////////

#include <Arduino.h>
#include <Wire.h>
#include <vl53l4cd_class.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define DEV_I2C Wire

// pin declarations
const int ledpin =  LED_BUILTIN;
const int led_micro =  3;
const int led_macro =  A3;
const int xshut_pin = A2;  // XSHUT pin for VL53L4CD

String device_name = "Libs internal controller";
String device_version = "v2";

// VL53L4CD distance sensor
VL53L4CD sensor_vl53l4cd_sat(&DEV_I2C, xshut_pin);
bool sensor_powered = false;

// Initialize (or reinitialize) the VL53L4CD sensor
void initSensor() {
  DEV_I2C.begin();
  sensor_vl53l4cd_sat.begin();
  sensor_vl53l4cd_sat.VL53L4CD_Off();
  sensor_vl53l4cd_sat.InitSensor();
  sensor_vl53l4cd_sat.VL53L4CD_SetRangeTiming(200, 0);
}

//----------------------------------SETUP------------------------------------

void setup() {

  pinMode(ledpin, OUTPUT);
  pinMode(led_micro, OUTPUT);
  pinMode(led_macro, OUTPUT);
  digitalWrite(ledpin, 0);
  digitalWrite(led_micro, 0);
  digitalWrite(led_macro, 0);

  Serial.begin(115200);

  // Initialize I2C bus
  DEV_I2C.begin();

  // Put sensor in standby at startup (XSHUT low)
  sensor_vl53l4cd_sat.VL53L4CD_Off();
}


void loop() {

  // check if data has been sent from the computer:
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();  // remove any trailing \r or spaces

        if (command == "?") {
        Serial.print(device_name);
        Serial.print(",");
        Serial.print(device_version);
        Serial.print("\r\n");
        delay(100);
        }

        else if (command == "H") {
        Serial.print("?                returns ID,version");
        Serial.print("\r\n");
        Serial.print("MICRO_LED_ON     turn LED micro ON");
        Serial.print("\r\n");
        Serial.print("MICRO_LED_OFF    turn LED micro OFF");
        Serial.print("\r\n");
        Serial.print("MACRO_LED_ON     turn LED macro ON");
        Serial.print("\r\n");
        Serial.print("MACRO_LED_OFF    turn LED macro OFF");
        Serial.print("\r\n");
        Serial.print("SENSOR_ON        wake up + init distance sensor");
        Serial.print("\r\n");
        Serial.print("SENSOR_OFF       put distance sensor in standby");
        Serial.print("\r\n");
        Serial.print("DIST             distance measurement (mm)");
        Serial.print("\r\n");
        Serial.print("DIST_DIAG        distance measurement (verbose)");
        Serial.print("\r\n");
        delay(100);
        }

        else if (command == "ON") {
        digitalWrite(led_micro, 1);
        Serial.print("led micro ON");
        Serial.print("\r\n");
        }

        else if (command == "OFF") {
        digitalWrite(led_micro, 0);
        Serial.print("led micro OFF");
        Serial.print("\r\n");
        }

        else if (command == "MACRO_LED_ON") {
        digitalWrite(led_macro, 1);
        Serial.print("led macro ON");
        Serial.print("\r\n");
        }

        else if (command == "MACRO_LED_OFF") {
        digitalWrite(led_macro, 0);
        Serial.print("led macro OFF");
        Serial.print("\r\n");
        }

        else if (command == "SENSOR_ON") {
        initSensor();
        sensor_powered = true;
        Serial.print("distance sensor ON");
        Serial.print("\r\n");
        }

        else if (command == "SENSOR_OFF") {
        sensor_vl53l4cd_sat.VL53L4CD_Off();  // XSHUT low = hardware standby
        sensor_powered = false;
        Serial.print("distance sensor OFF");
        Serial.print("\r\n");
        }

        else if (command == "DIST" || command == "DIST_DIAG") {
        if (!sensor_powered) {
          Serial.println("ERROR: sensor is OFF, send SENSOR_ON first");
        } else {
          // Trigger a distance measurement using VL53L4CD
          uint8_t NewDataReady = 0;
          VL53L4CD_Result_t results;
          uint8_t status;

          // Start ranging
          sensor_vl53l4cd_sat.VL53L4CD_StartRanging();

          // Wait for new data to be ready
          do {
            status = sensor_vl53l4cd_sat.VL53L4CD_CheckForDataReady(&NewDataReady);
          } while (!NewDataReady);

          if ((!status) && (NewDataReady != 0)) {
            // Clear HW interrupt to restart measurements
            sensor_vl53l4cd_sat.VL53L4CD_ClearInterrupt();

            // Read measured distance. RangeStatus = 0 means valid data
            sensor_vl53l4cd_sat.VL53L4CD_GetResult(&results);

            if (command == "DIST_DIAG") {
              char report[64];
              snprintf(report, sizeof(report), "Status = %3u, Distance = %5u mm, Signal = %6u kcps/spad",
                       results.range_status,
                       results.distance_mm,
                       results.signal_per_spad_kcps);
              Serial.println(report);
            } else {
              Serial.println(results.distance_mm);
            }
          } else {
            Serial.println("ERROR: measurement failed");
          }

          // Stop ranging
          sensor_vl53l4cd_sat.VL53L4CD_StopRanging();
        }
        }

  }
}
