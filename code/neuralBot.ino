/******************************************************************
 * neuralBot.ino
 * Jaume Nogués, rPrimtech
 * 
 * neuralBot, a simple neural network robot
 * 
 * Powered by ESP32
 * Souce code in:
 * https://github.com/jnogues/neuralBot
 * 
 * Copyright (C) 2016-2018 by Jaume Nogués <jnogues at gmail dot com>
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * 
 * Inspired by:
 * http://robotics.hobbizine.com/arduinoann.html
 * https://github.com/IdleHandsProject/makennbot
 * 
 * 
 ******************************************************************/

#include <math.h>
#include <Arduino.h>
#include <EEPROM.h>
#include <Wire.h>
#include <VL53L0X.h>      //https://github.com/pololu/vl53l0x-arduino
#include "SSD1306.h"      //https://github.com/squix78/esp8266-oled-ssd1306
#include <ESP32_Servo.h>  //https://github.com/jkb-git/ESP32Servo

SSD1306  display(0x3c, 21, 22);
VL53L0X sensorL;
VL53L0X sensorR;
#define HIGH_SPEED
//#define HIGH_ACCURACY

Servo servoLeft;
Servo servoRight;
#include <SimpleTimer.h> //https://github.com/jfturcot/SimpleTimer
SimpleTimer timer;

/******************************************************************
 * Network Configuration - customized per network 
 * All basic settings can be controlled via the Network Configuration
 ******************************************************************/

const int PatternCount = 31;
const int InputNodes = 2;
const int HiddenNodes = 10;
const int OutputNodes = 2;
const float LearningRate = 0.1;//0.3
const float Momentum = 0.1;//0.9
const float InitialWeightMax = 0.5;
const float Success = 0.04;//0.004;

float Input[PatternCount][InputNodes] = {
  {1.0, 1.0},   // 30 30
  {1.0, 0.833}, // 30 25
  {0.833, 1.0}, // 25 30
  {0.7, 0.933}, // 21 28
  {0.933, 0.7}, // 28 21
  {0.733, 0.8}, // 22 24
  {0.8, 0.733}, // 24 22
  {0.733, 0.7}, // 22 21
  {0.7, 0.733}, // 21 22

  //{0.666, 0.633},  // 20 19
  {0.666, 0.6},    // 20 18
  //{0.6, 0.533},    // 18 16
  {0.533, 0.466},  // 16 14
  //{0.466, 0.4},    // 14 12
  {0.4, 0.333},    // 12 10

  //{0.633, 0.666},  // 19 20
  {0.6, 0.666},    // 18 20
  //{0.533, 0.6},    // 16 18
  {0.466, 0.533},  // 14 16
  //{0.4, 0.466},    // 12 14
  {0.333, 0.4},    // 10 12

  {0.333, 0.333}, // 10 10 
  {0.266, 0.333}, // 8 10 
  {0.333, 0.266}, // 10 8 
  {0.2, 0.266},   // 6 8 
  {0.266, 0.2},   // 8 6 
  {0.166, 0.166}, //5 5
  {1.0, 0.266},   //30 8 
  {0.266, 1},     //8 30
  {0.933, 0.2},   //28 6
  {0.2, 0.933},   //6 28

  {1.0, 0.6},   // 30 18
  {0.933, 0.5}, // 28 15
  {0.86, 0.4},  // 26 12

  {0.6, 1.0},   // 18 30
  {0.5, 0.933}, // 15 28
  {0.4, 0.86},  // 12 26  

}; 

const byte Target[PatternCount][OutputNodes] = {
  { 1, 1},  
  { 1, 1},
  { 1, 1},
  { 1, 1},
  { 1, 1},
  { 1, 1},
  { 1, 1},
  { 1, 1},
  { 1, 1},

  //{ 0, 1},
  { 0, 1},
  //{ 0, 1},
  { 0, 1},
  //{ 0, 1},
  { 0, 1}, 

  //{ 1, 0},
  { 1, 0},
  //{ 1, 0},
  { 1, 0},
  //{ 1, 0},
  { 1, 0},

  { 0, 0},
  { 0, 0},
  { 0, 0},
  { 0, 0},
  { 0, 0},
  { 0, 0},
  { 0, 0},
  { 0, 0},
  { 0, 0},
  { 0, 0},

  { 0, 1},
  { 0, 1},
  { 0, 1},

  { 1, 0},
  { 1, 0},
  { 1, 0},  

};

/******************************************************************
 * End Network Configuration
 ******************************************************************/

//init variables
int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][OutputNodes];

boolean bOutput[OutputNodes];
unsigned int neuralLoopsCounter=0;

int countEEPROM=0; //eeprom counter
const int eepromHiddenWeights=0; // first eeprom position for save HiddenWeights
const int eepromOutputWeights=512; // first eeprom position for save OutputWeights

unsigned int distanceL=0;
unsigned int distanceR=0;
float distanceL_float=0;
float distanceR_float=0;
int speedLfoward=1070;//1100
int speedLreverse=2000;
int speedRfoward=1950;//1950
int speedRreverse=1000;

boolean T4now=0;
boolean T4past=0;
boolean ON=0;
byte touchLimit=60;

void setup()
{
  pinMode(16,OUTPUT);
  pinMode(17,OUTPUT);
  pinMode(2,OUTPUT);
  digitalWrite(16, LOW);
  digitalWrite(17, LOW);
  digitalWrite(2, LOW);
  servoLeft.attach(15);
  servoRight.attach(12);
  
  EEPROM.begin(4096);  //EEPROM.begin(Size)
  
  Serial.begin(115200);
  Serial.println("Starting....");

  display.init();
  //display.flipScreenVertically();
  display.setFont(ArialMT_Plain_16);// ArialMT_Plain_10, ArialMT_Plain_16, ArialMT_Plain_24
  display.drawString(0, 0,  "   NEURAL BOT");
  display.drawString(0, 24, "   PUSH START");
  display.display();

  Wire.begin();//call after display init and config
  digitalWrite(16, HIGH);
  delay(100);
  sensorL.init();
  sensorL.setAddress(0x33);
  sensorL.setTimeout(200);
  
  digitalWrite(17, HIGH);
  delay(100);
  sensorR.init();
  sensorR.setAddress(0x34);
  sensorR.setTimeout(200);
  
  #if defined LONG_RANGE
  // lower the return signal rate limit (default is 0.25 MCPS)
  sensorL.setSignalRateLimit(0.1);
  sensorR.setSignalRateLimit(0.1);
  // increase laser pulse periods (defaults are 14 and 10 PCLKs)
  sensorL.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 18);
  sensorL.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 14);
  sensorR.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 18);
  sensorR.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 14);
  #endif

  #if defined HIGH_SPEED
  // reduce timing budget to 20 ms (default is about 33 ms)
  sensorL.setMeasurementTimingBudget(20000);
  sensorR.setMeasurementTimingBudget(20000);
  #elif defined HIGH_ACCURACY
  // increase timing budget to 200 ms
  sensorL.setMeasurementTimingBudget(200000);
  sensorR.setMeasurementTimingBudget(200000);
  #endif

  timer.setInterval(100, checkButtons);
  timer.setInterval(200, showDistances);
  
  //trainNeuralNetwork();saveWeights2eeprom(); //unncomment for train network and save results
  
  readWeights2eeprom(); 
}  


void loop ()
{
      timer.run();
      if (ON)
      {
        readDistance();
        InputToOutput(distanceL_float, distanceR_float);
      }  
}



  /*****************************************************************
      Functions
  *****************************************************************/

void InputToOutput(float In1, float In2)
{
  unsigned long neuralTime=micros();
  float TestInput[] = {0, 0};
  TestInput[0] = In1;
  TestInput[1] = In2;

  /******************************************************************
    Compute hidden layer activations
  ******************************************************************/

  for ( i = 0 ; i < HiddenNodes ; i++ ) {
    Accum = HiddenWeights[InputNodes][i] ;
    //Serial.print("Accum1="); Serial.println(Accum);//show bias
    for ( j = 0 ; j < InputNodes ; j++ ) {
      Accum += TestInput[j] * HiddenWeights[j][i] ;
    }
    Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;
  }

  /******************************************************************
    Compute output layer activations and calculate errors
  ******************************************************************/

  for ( i = 0 ; i < OutputNodes ; i++ ) {
    Accum = OutputWeights[HiddenNodes][i] ;
    //Serial.print("Accum2="); Serial.println(Accum);//show bias
    for ( j = 0 ; j < HiddenNodes ; j++ ) {
      Accum += Hidden[j] * OutputWeights[j][i] ;
    }
    Output[i] = 1.0 / (1.0 + exp(-Accum)) ;

    if (Output[i]>=0.5) bOutput[i]= 1;
    if (Output[i] <0.5) bOutput[i]= 0;
    //Serial.print(bOutput[i]);
    //Serial.print(" ");
  }
  //Serial.println(" ");

  if (bOutput[0]==1) {servoLeft.writeMicroseconds(speedLfoward);}//foward
  if (bOutput[0]==0) {servoLeft.writeMicroseconds(speedLreverse);}//reverse
  if (bOutput[1]==1) {servoRight.writeMicroseconds(speedRfoward);}//foward
  if (bOutput[1]==0) {servoRight.writeMicroseconds(speedRreverse);}//reverse

  //every 1000 loops in neural network, built in led toggle
  neuralLoopsCounter=neuralLoopsCounter+1;
  if (neuralLoopsCounter==100)//1000
  {
    digitalWrite(2,!digitalRead(2));
    neuralLoopsCounter=0;
  }
  //Serial.println(micros()-neuralTime);
}


void toTerminal()
{

  for( p = 0 ; p < PatternCount ; p++ ) { 
    Serial.println(); 
    Serial.print ("  Training Pattern: ");
    Serial.println (p);      
    Serial.print ("  Input ");
    for( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Target ");
    for( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }
/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
        Serial.print (HiddenWeights[j][i],1);
        Serial.print (" ");
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
        Serial.print (OutputWeights[j][i],1);
        Serial.print (" ");
      }
      Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.print ("  Output ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
  }
}


void saveWeights2eeprom()
{
  /******************************************************************
* Save hidden weights
******************************************************************/
    Serial.println("Saving Weights!");
    countEEPROM=eepromHiddenWeights;
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      //Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j <= InputNodes ; j++ ) { //very important <=
        //Accum += Input[p][j] * HiddenWeights[j][i] ;
        Serial.print (HiddenWeights[j][i],5);
        Serial.print (" ");
        EEPROM.put(countEEPROM,HiddenWeights[j][i]); // Writes HiddenWeights[j][i] to EEPROM
        EEPROM.commit();
        countEEPROM=countEEPROM+4;
      }
      //Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }
    Serial.println(" ");
    Serial.print("last eeprom byte=");
    Serial.println(countEEPROM);
/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/
    countEEPROM=eepromOutputWeights;
    for( i = 0 ; i < OutputNodes ; i++ ) {    
      //Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j <= HiddenNodes ; j++ ) { //very important <=
        //Accum += Hidden[j] * OutputWeights[j][i] ;
        Serial.print (OutputWeights[j][i],5);
        Serial.print (" ");
        EEPROM.put(countEEPROM,OutputWeights[j][i]); // Writes OutputWeights[j][i] to EEPROM
        EEPROM.commit();
        countEEPROM=countEEPROM+4;
      }
      //Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.print ("  Output ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
    Serial.println(" ");
    Serial.print("last eeprom byte=");
    Serial.println(countEEPROM);
    Serial.println("Saved!");
}

void readWeights2eeprom()
{
  /******************************************************************
* Read hidden weig
******************************************************************/
    Serial.println("Reading Weights!");
    countEEPROM=eepromHiddenWeights;
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      //Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j <= InputNodes ; j++ ) { //very important <=
        //Accum += Input[p][j] * HiddenWeights[j][i] ;
        EEPROM.get(countEEPROM,HiddenWeights[j][i]);
        Serial.print (HiddenWeights[j][i],5);
        Serial.print (" ");
        countEEPROM=countEEPROM+4;
      }
    }
    Serial.println(" ");
    Serial.println("End read HiddenWeights");
/******************************************************************
* Read output layer activations and calculate errors
******************************************************************/
    countEEPROM=eepromOutputWeights;
    for( i = 0 ; i < OutputNodes ; i++ ) {    
      //Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j <= HiddenNodes ; j++ ) { //very important <=
        //Accum += Hidden[j] * OutputWeights[j][i] ;
        EEPROM.get(countEEPROM,OutputWeights[j][i]);
        Serial.print (OutputWeights[j][i],5);
        Serial.print (" ");
        countEEPROM=countEEPROM+4;
      }
      //Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.println(" ");
    Serial.println("Readed!");
}

void trainNeuralNetwork()
{  
  display.clear();
  display.drawString(0, 0, "Training...");
  display.display();
  randomSeed(analogRead(0)); //Collect a random ADC sample for Randomization.
  ReportEvery1000 = 1;
  for( p = 0 ; p < PatternCount ; p++ ) {    
    RandomizedIndex[p] = p ;
  }
/******************************************************************
* Initialize HiddenWeights and ChangeHiddenWeights 
******************************************************************/

  for( i = 0 ; i < HiddenNodes ; i++ ) {    
    for( j = 0 ; j <= InputNodes ; j++ ) { 
      yield();
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100))/100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
/******************************************************************
* Initialize OutputWeights and ChangeOutputWeights
******************************************************************/

  for( i = 0 ; i < OutputNodes ; i ++ ) {    
    for( j = 0 ; j <= HiddenNodes ; j++ ) {
      yield();
      ChangeOutputWeights[j][i] = 0.0 ;  
      Rando = float(random(100))/100;        
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
  Serial.println("Initial/Untrained Outputs: ");
  toTerminal();
/******************************************************************
* Begin training 
******************************************************************/

  for( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {    

/******************************************************************
* Randomize order of training patterns
******************************************************************/

    for( p = 0 ; p < PatternCount ; p++) {
      yield();
      q = random(PatternCount);
      r = RandomizedIndex[p] ; 
      RandomizedIndex[p] = RandomizedIndex[q] ; 
      RandomizedIndex[q] = r ;
    }
    Error = 0.0 ;
/******************************************************************
* Cycle through each training pattern in the randomized order
******************************************************************/
    for( q = 0 ; q < PatternCount ; q++ ) {    
      p = RandomizedIndex[q];

/******************************************************************
* Compute hidden layer activations
******************************************************************/

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = HiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
      }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

      for( i = 0 ; i < OutputNodes ; i++ ) {    
        Accum = OutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }

/******************************************************************
* Backpropagate errors to hidden layer
******************************************************************/

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = 0.0 ;
        for( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }


/******************************************************************
* Update Inner-->Hidden Weights
******************************************************************/


      for( i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) { 
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

/******************************************************************
* Update Hidden-->Output Weights
******************************************************************/

      for( i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

/******************************************************************
* Every 1000 cycles send data to terminal for display
******************************************************************/
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0)
    {
      Serial.println(); 
      Serial.println(); 
      Serial.print ("TrainingCycle: ");
      Serial.print (TrainingCycle);
      Serial.print ("  Error = ");
      Serial.println (Error, 5);
      yield();
      digitalWrite(2,!digitalRead(2));
      toTerminal();

      if (TrainingCycle==1)
      {
        ReportEvery1000 = 999;
      }
      else
      {
        ReportEvery1000 = 1000;
      }
    }    


/******************************************************************
* If error rate is less than pre-determined threshold then end
******************************************************************/

    if( Error < Success ) break ;  
  }
  Serial.println ();
  Serial.println(); 
  Serial.print ("TrainingCycle: ");
  Serial.print (TrainingCycle);
  Serial.print ("  Error = ");
  Serial.println (Error, 5);

  toTerminal();

  Serial.println ();  
  Serial.println ();
  Serial.println ("Training Set Solved! ");
  Serial.println ("--------"); 
  Serial.println ();
  Serial.println ();  
  ReportEvery1000 = 1;
  Serial.println(millis());
  display.drawString(0, 0, "End Train");
  display.display();
}


void readDistance()
{
  distanceL=sensorL.readRangeSingleMillimeters();
  distanceR=sensorR.readRangeSingleMillimeters();
  if (distanceL>=300) {distanceL=300;}
  if (distanceR>=300) {distanceR=300;}
  distanceL_float= distanceL * (1.0/300);
  distanceR_float= distanceR * (1.0/300);
}

void checkButtons()
{
  byte touchValue=touchRead(T4)+touchRead(T4)+touchRead(T4);
  //Serial.println(touchValue);
  if ((touchValue<=touchLimit) && T4past==0) 
  {
        ON=!ON; 
        T4past=1;
        if (ON) 
        {
          display.clear();
          display.drawString(0, 18, "    STARTING....");
          display.display();
          delay(1000);        
        }
        if (!ON) 
        {
          servoLeft.write(90); 
          servoRight.write(90);
          display.clear();
          display.drawString(0, 18, "       STOPED    ");
          display.display();
          delay(500);
        }
   }     
   if ((touchValue>40) && T4past==1) { T4past=0;}
      
}

void showDistances()
{
  if (ON)
  {
  display.clear();
  display.drawString(30, 15, "RUNNING");
  String message="L = "+String(distanceL);
  display.drawString(0, 42, message);
  message="R = "+String(distanceR);
  display.drawString(68, 42, message);
  display.display();
  }
}
