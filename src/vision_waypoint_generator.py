#!/usr/bin/env python
# =============== import library ======================
from __future__ import print_function, division
from std_msgs.msg import Float32MultiArray,MultiArrayDimension
from sensor_msgs.msg import Image,CameraInfo
from autoware_msgs.msg import PointsImage
from geometry_msgs.msg import PoseArray , Pose , PoseStamped
from autoware_msgs.msg import LaneArray,Lane,Waypoint

import rospkg
import rospy
import tf

from cv_bridge import CvBridge
import time

import cv2
import numpy as np
import os
import sys

import matplotlib.pylab as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid

segment = 20
delta_y = 20
main_point_raw = []




'''
=============== Subscribe  /mainpoint ======================
* Subscribe /points_image
* Subscribe /main_point
* Subscribe /points_image
* Subscribe /points_image
* Subscribe /Drive/main_point
* Subscribe /camera_info
* Subscribe /current_pose
'''
class Img_Sub():
    def __init__(self):
        self.points_image_sub = rospy.Subscriber("/points_image", PointsImage, self.callback)
        self.points_image_ok = False
        self.distance = np.zeros((576, 1024)).astype(np.float)
        self.intensity = np.ones((576, 1024)).astype(np.float)
    def callback(self, msg):

        image_height,image_width = msg.image_height,msg.image_width
        self.distance = np.array(msg.distance).reshape((image_height,image_width))
        
        self.points_image_ok = True

class Main_Point_Sub():
    def __init__(self):
        self.mainpoint_sub = rospy.Subscriber("Drive/main_point", Float32MultiArray,self.callback)
    def callback(self, msg):
        global main_point_raw
        main_point_raw = msg.data

class CameraInfo_Sub():
    def __init__(self,camera_info_topic):
        self.camera_sub = rospy.Subscriber(camera_info_topic, CameraInfo,self.callback)
        self.F_intrc_inv =  np.zeros((3,3))
        self.camera_ok = False
    def callback(self, msg):
        F_intrc = np.array(msg.P)
        F_intrc = F_intrc.reshape((3,4))[:,:3]
        F_intrc_inv = np.linalg.inv(F_intrc)
        self.F_intrc_inv = F_intrc_inv
        self.camera_ok = True

class CurrentPose_Sub():
    def __init__(self):
        self.pose_sub = rospy.Subscriber("/current_pose", PoseStamped ,self.callback)
    def callback(self, msg):
        # A Pose with reference coordinate frame and timestamp
        #Header header
        #Pose pose
        x ,y ,z, w = msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w
        
        r,p,y = tf.transformations.euler_from_quaternion((x,y,z,w))





        
class tf_listen_class():
    def __init__(self,camera_frame):
        self.listener = tf.TransformListener()
        self.camera_frame = camera_frame
        now = rospy.Time.now()
        self.trans = [0,0,0]
        self.rot = [0,0,0,0]
        self.projection_matrix = np.zeros((4,4))
        
    def wait_update_transform(self):
        t = self.listener.waitForTransform('/map',self.camera_frame,rospy.Time.now(), rospy.Duration(500.0))
        (trans, rot) = self.listener.lookupTransform('/map', self.camera_frame, rospy.Time())
        
        # update rot trans matrix
        self.trans = trans
        self.rot = rot
        self.projection_matrix = self.listener.fromTranslationRotation(trans,rot)

       
    def transformPose(self,points):
        lane_array = LaneArray()
        lane = Lane()
        lane.header.frame_id='map'
        
        
        points = np.array(points)
        points_in_map = np.dot(self.projection_matrix,points.transpose()).transpose()
        for i in range(len(points)):
            waypoint = Waypoint()
            p1 = PoseStamped()
            p1.header.frame_id = 'map'
            p1.pose.position.x,p1.pose.position.y,p1.pose.position.z = points_in_map[i][0],points_in_map[i][1],points_in_map[i][2]
            if i != len(points)-1:
                dif_x,dif_y,dif_z = points_in_map[i+1][0]-points_in_map[i][0],points_in_map[i+1][1]-points_in_map[i][1],points_in_map[i+1][2]-points_in_map[i][2]
                yaw = np.angle([dif_x+dif_y*1j])
                quaternion = tf.transformations.quaternion_from_euler(0,0,yaw)

                p1.pose.orientation.x,p1.pose.orientation.y,p1.pose.orientation.z,p1.pose.orientation.w = quaternion
                
            else:
                p1.pose.orientation.x,p1.pose.orientation.y,p1.pose.orientation.z,p1.pose.orientation.w = 0,0,0,1 #forward
            waypoint.pose = p1
            waypoint.twist.twist.linear.x =1.
            lane.waypoints.append(waypoint)

        lane_array.lanes.append(lane)
        return lane_array





# =============== Node Init  ======================
rospy.init_node('vision_waypoint_generator', anonymous=True)

# =============== ROS Param Read  ======================

camera_info_topic = rospy.get_param('camera_info_topic','/usb_camc/camera_info')
camera_frame = rospy.get_param('camera_frame','/camerac')
pub_rate = rospy.get_param('rate',1)

eps = rospy.get_param('eps',1)
min_samples = rospy.get_param('min_samples',3)
line_width = rospy.get_param('line_width',15)

# =============== Subscribe Node Init  ======================
points_image_sub_node = Img_Sub()
main_point_sub_node = Main_Point_Sub()
camerainfo_sub_node  = CameraInfo_Sub(camera_info_topic)
current_pose_sub_node = CurrentPose_Sub()

# =============== TF Transform Node  ======================
tf_transform_camera_map = tf_listen_class(camera_frame)

# =============== Publisher Init  ======================
posearray_pub_seg = rospy.Publisher("/lane_waypoints_array",LaneArray )


rate = rospy.Rate(pub_rate)
plt.ion()
f,ax = plt.subplots(1,2,figsize=(16,4))
while not rospy.is_shutdown():
    
    if not (camerainfo_sub_node.camera_ok and points_image_sub_node.points_image_ok):
        print("Vision_Waypoint_Generator: camera info / Current Pose not ready")
        time.sleep(0.2)
        continue
    F_intrc_inv = camerainfo_sub_node.F_intrc_inv
    if not len(main_point_raw):
        print('Not Enough main pint raw : skipping')
        continue

    main_point = np.array(main_point_raw).reshape(-1,2).astype(np.int32)
    
    distance  = points_image_sub_node.distance.copy()
    poly_line_mask = np.zeros(distance.shape,dtype=np.uint8)
    cv2.polylines(poly_line_mask, [main_point.reshape(-1,1,2)], False, 1, line_width)
    y,x = np.nonzero(distance.copy() * poly_line_mask)
    
    # No Lidar Point in path  ---------------
    if len(x)==0:
        #     points = np.array(points)
        print('No Lidar In Path')
        posearray_pub_seg.publish(LaneArray())
        continue
    
    py,px = np.nonzero(distance)
    lidar_in_path = np.stack((y,x,distance[y,x]/100.0)).T.astype(int)
    
    
    
    points = np.stack((x*distance[y,x]/100.0,y*distance[y,x]/100.0,distance[y,x]/100.0)).T.astype(int)
    transform_points = np.ones((lidar_in_path.shape[0],4),dtype=np.float32)
    
    transform_points[:,:3] = np.dot(F_intrc_inv,points.T).T
    
    
    # No Cluster Founded  ---------------
    db = DBSCAN(eps=1, min_samples=3).fit(transform_points[:,:3])
    c = db.labels_
    
    if np.all(c<1):
        print("No clister find")
        posearray_pub_seg.publish(LaneArray())
        continue
    
    
    clf = NearestCentroid()
    clf.fit(transform_points, c)
    centroids = clf.centroids_
    
    # remove -1 class 
    if -1 in c:
        centroids= np.delete(centroids,(0),axis=0)
    #sort by distance decrease order
    centroids = centroids[centroids[:,2].argsort()[::-1]]
    
    
    
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(poly_line_mask)
    ax[0].scatter(x,y,s=1,c=c,cmap=plt.cm.cool)
    ax[0].set_title("Planning Result")
    ax[1].imshow(poly_line_mask)
    ax[1].scatter(px,py,s=1,c=distance[py,px],cmap=plt.cm.cool)
    ax[1].set_title("Planning Result Overlay with lidar")

    f.canvas.draw()
    
    

    tf_transform_camera_map.wait_update_transform()
    print('update')
    posearray = tf_transform_camera_map.transformPose(centroids[::-1])
    posearray_pub_seg.publish(posearray)
    rate.sleep()
