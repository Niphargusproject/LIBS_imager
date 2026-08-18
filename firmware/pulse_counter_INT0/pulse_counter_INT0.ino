#include <SerialCommand.h>   /* http://github.com/p-v-o-s/Arduino-SerialCommand */

//Number of pulses, used to measure energy.
volatile long pulseCount = 0;
volatile unsigned long trigger_emit_count = 0;
int first_pulse = 0;

//Used to measure power.
unsigned long pulseTime,lastTime;

//power and energy
double timer;
const int trigger_pin = 4 ;
volatile int step_target=1;

// Non-blocking trigger: ISR only sets pin HIGH; loop() turns it LOW after 100 us.
// Avoids missing step pulses when we used delayMicroseconds() inside the ISR.
volatile unsigned long trigger_off_at_micros = 0;  // 0 = no pending turn-off
const unsigned int trigger_pulse_us = 100;

SerialCommand sCmd(Serial);// the SerialCommand parser object

void setup()
{
    Serial.begin(115200);
    while (!Serial) {    ; // wait for serial port to connect. Needed for native USB port only
    }
    pinMode(trigger_pin, OUTPUT) ;
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(2, INPUT);
    digitalWrite(trigger_pin, 0);
    // KWH interrupt attached to IRQ 0 = pin2

sCmd.setDefaultHandler(UNRECOGNIZED_sCmd_default_handler);
sCmd.addCommand("STEP", STEP_sCmd_config_handler);     //configures integration time
sCmd.addCommand("START", START_sCmd_query_handler);       //reads out the whole spectrum
sCmd.addCommand("STOP", STOP_sCmd_query_handler);   //reads out timings
sCmd.addCommand("COUNT", COUNT_sCmd_query_handler);   //reads out emitted trigger count
sCmd.addCommand("RESETCOUNT", RESETCOUNT_sCmd_query_handler);   //resets emitted trigger count
sCmd.addCommand("FILTER", STOP_sCmd_query_handler);   //reads out timings
sCmd.addCommand("PUMP_ON", STOP_sCmd_query_handler);   //reads out timings
sCmd.addCommand("PUMP_OFF", STOP_sCmd_query_handler);   //reads out timings
sCmd.addCommand("H", HELP_sCmd_query_handler);   //reads out timings
sCmd.addCommand("?", IDENTITY_sCmd_query_handler);   //reads out timings

}

void loop()
{
  // Turn trigger pin LOW after 100 us (non-blocking; keeps ISR short so we don't miss steps).
  // Read trigger_off_at_micros with interrupts off (4-byte read is not atomic on AVR).
  noInterrupts();
  unsigned long t_off = trigger_off_at_micros;
  interrupts();
  if (t_off != 0 && (long)(micros() - t_off) >= 0) {
    digitalWrite(trigger_pin, 0);
    trigger_off_at_micros = 0;
  }

  int num_bytes = sCmd.readSerial();      // fill the buffer
  if (num_bytes > 0){
    sCmd.processCommand();  // process the command
  }
}

// Unrecognized command
void UNRECOGNIZED_sCmd_default_handler(SerialCommand this_sCmd)
{
  SerialCommand::CommandInfo command = this_sCmd.getCurrentCommand();
  this_sCmd.print(F("### Error: command '"));
  this_sCmd.print(command.name);
  this_sCmd.print(F("' not recognized ###\n"));
}

void STEP_sCmd_config_handler(SerialCommand this_sCmd){
  char *arg = this_sCmd.next();
  if (arg == NULL){
    this_sCmd.println(F("### Error: STEP command requires 1 argument 'STEP NUMBER'"));
  }
  else{
    step_target = atoi(arg);
    this_sCmd.print(F("STEP : "));
    this_sCmd.println(step_target);
  }
}

void START_sCmd_query_handler(SerialCommand this_sCmd){
  attachInterrupt(digitalPinToInterrupt(2), onPulse, RISING);
  pulseCount = 0;
  first_pulse = 0;
  this_sCmd.println(F("START : START pulsing with external interrupt"));
  
}
void STOP_sCmd_query_handler(SerialCommand this_sCmd){
  detachInterrupt(digitalPinToInterrupt(2));
  this_sCmd.println(F("STOP pulsing with external interrupt"));
}

void COUNT_sCmd_query_handler(SerialCommand this_sCmd){
  noInterrupts();
  unsigned long count = trigger_emit_count;
  interrupts();
  this_sCmd.print(F("COUNT : "));
  this_sCmd.println(count);
}

void RESETCOUNT_sCmd_query_handler(SerialCommand this_sCmd){
  noInterrupts();
  trigger_emit_count = 0;
  pulseCount = 0;
  interrupts();
  this_sCmd.println(F("RESETCOUNT : 0"));
}

void HELP_sCmd_query_handler(SerialCommand this_sCmd){
  this_sCmd.println(F("STEP : define number of motor steps or microsteps befor laser pulse"));
  this_sCmd.println(F("START : START pulsing with external interrupt"));
  this_sCmd.println(F("STOP : STOP pulsing with external interrupt"));
  this_sCmd.println(F("COUNT : read emitted trigger count"));
  this_sCmd.println(F("RESETCOUNT : reset emitted trigger count"));
  this_sCmd.println(F("FILTER : TOGGLE filter machine"));
  this_sCmd.println(F("PUMP_ON : Start the pump"));
  this_sCmd.println(F("PUMP_OFF : Stop the pump"));  
  this_sCmd.println(F("H : list commands"));
  this_sCmd.println(F("? : identify device"));
}

void IDENTITY_sCmd_query_handler(SerialCommand this_sCmd){
  this_sCmd.println(F("LIBS motor pulse counter v1.1"));
}


// The interrupt routine — keep it minimal: no delayMicroseconds() here or we miss step pulses.
void onPulse()
{
  pulseCount++;

  if (pulseCount == step_target) {
    digitalWrite(trigger_pin, 1);
    trigger_off_at_micros = micros() + trigger_pulse_us;  // loop() will turn off after 100 us
    trigger_emit_count++;
    pulseCount = 0;
  }
}
