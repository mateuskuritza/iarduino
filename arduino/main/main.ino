const int ledPins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
const int numPins = sizeof(ledPins) / sizeof(ledPins[0]);

void setup()
{
    Serial.begin(9600);

    for (int i = 0; i < numPins; i++)
    {
        pinMode(ledPins[i], OUTPUT);
    }
}

void loop()
{
    if (Serial.available())
    {
        String command = Serial.readStringUntil('\n');

        if (command.startsWith("LED:"))
        {
            int pin = command.substring(4).toInt();

            for (int i = 0; i < numPins; i++)
            {
                if (ledPins[i] == pin)
                {
                    digitalWrite(pin, HIGH);
                }
                else
                {
                    digitalWrite(ledPins[i], LOW);
                }
            }
        }
    }
}
