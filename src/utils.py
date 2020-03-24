import random

def randomString(length, alreadyTook = []):
    str = None
    while (str in alreadyTook) or (str == None):
        str = ''
        for i in range(0, length):
            str = str + chr(random.randint(97, 98))
    return str