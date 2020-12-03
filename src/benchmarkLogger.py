import threading
import time
import psutil
import csv

class benchThread (threading.Thread):
   def __init__(self, pollTime, active):
      threading.Thread.__init__(self)
      self.pollTime = pollTime
      self.active = active

   def run(self):
      timeCounter = 0
      print ("Starting " + self.name)
      with open('innovators.csv', 'w+', newline='') as file:
         writer = csv.writer(file)
         writer.writerow(["Time", "CPU %", "Memory"])
      while self.active:
         timeCounter += self.pollTime
         print_time(self.name, self.pollTime, timeCounter)
      print ("Exiting " + self.name)

def print_time(threadName, pollTime, timeCounter):
   time.sleep(pollTime)
   #print ("%s: %s %.2f" % (threadName, psutil.cpu_percent(), psutil.virtual_memory().used/(1024**2)))
   with open('innovators.csv', 'a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([timeCounter, psutil.cpu_percent(), psutil.virtual_memory().used/(1024**2)])


# Create new threads

#thread1 = benchThread(1, "Thread-1", 1, 1)
#thread2 = benchThread(2, "Thread-2", 2, 1)
#thread3 = threading.Thread(target=print_time("pesho", 3, 4, 1))

# Start new Threads
#thread1.start()
#thread2.start()
#thread1.join()
#thread2.join()
#thread3.start()
#thread3.join()
#time.sleep(2)
#thread1.active = 0
#print ("Exiting Main Thread")