
#include <Arduino.h>
#include <DFRobot_TactileSensor.h>

#define ADDRESS 1
#define ARRAY_SIZE 32

DFRobot_TactileSensor tactile(ADDRESS, &Serial1);

void setup() {
    Serial.begin(115200);
    Serial1.begin(115200);

    delay(1000);
    Serial.println("Starting SEN0705 test...");

    while (tactile.begin(ARRAY_SIZE) == -1) {
        Serial.println("Sensor not detected. Check wiring.");
        delay(1000);
    }

    Serial.println("Sensor connected!");
    tactile.setSampleRate(eSampleRate20Hz);
}

void loop() {
    sAdcDatas_t data = tactile.getDatas();

    if (data.result == 0) {
        for (int row = 0; row < tactileSensorArrayY; row++) {
            for (int col = 0; col < tactileSensorArrayX; col++) {
                Serial.print(data.adcval[row][col]);
                Serial.print("\t");
            }
            Serial.println();
        }
        Serial.println();
    } else {
        Serial.println("Sensor read failed");
    }

    delay(100);
}
