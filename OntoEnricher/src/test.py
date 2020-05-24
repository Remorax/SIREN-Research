import threading
import subprocess
import time
import re

def stats():
    t = threading.Timer(5.0, stats)
    t.daemon=True
    t.start()
    bashCommand1 = "ps -ef"
    process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output, error = process1.communicate()
    output = re.findall(".* python3.*", output.decode("utf-8"))
    print ("\n".join(output))

stats()


print (2+3)


time.sleep(6)

print ("boo")

time.sleep(6)
