import math

def conv_C_F(tempC):
    return tempC * 9 / 5 + 32


if __name__ == "__main__":

    tempC = 20 * (180 / math.pi)
    type(tempC)  # float

    tempF = tempC * 9 / 5 + 32  # Temperatur in Farad
    type(tempF)  # float
    print(conv_C_F(20))
