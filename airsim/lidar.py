import airsim
import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
 
# get control
client.enableApiControl(True)
 
# unlock
client.armDisarm(True)

time = 0.0

def parse_lidarData(data):
# reshape array of floats to array of [X,Y,Z]
    points = np.array(data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0]/3), 3))
   
    return points

# Async methods returns Future. Call join() to wait for task to complete.
fig = plt.figure()
ax = fig.gca(projection='3d')
client.takeoffAsync().join()

li = parse_lidarData(airsim.MultirotorClient().getLidarData())

for i in range(10):
    client.hoverAsync().join()
    raw = parse_lidarData(airsim.MultirotorClient().getLidarData())
    li = np.concatenate((li,raw),axis = 0)

li = li.transpose()
X = li[0]
Y = li[1]
Z = li[2]
ax.scatter(X,Y,Z,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')
plt.show()
print(li)
#while True:
#    vx = 10*m.sin(time)
#    vy = 10*m.cos(time)
#    time+=0.1
#    client.moveByVelocityAsync(vx, vy, 1,1).join()

#client.landAsync().join()

# lock
client.armDisarm(False)
# release control
client.enableApiControl(False)

#test1 zch