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
import pandas as pd
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
class Pred_Max_Sub():
    def __init__(self):
        self.pred_max_sub = rospy.Subscriber("Drive/pred_max", Image,self.callback)
        self.pred_max = np.zeros((576,1024,3),dtype=np.uint8)
        self.bridge = CvBridge()
        self.ok = False
    def callback(self, msg):
        self.pred_max = self.bridge.imgmsg_to_cv2(msg)
        self.ok = True

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
        #Pose pose (location on map frame )
        x ,y ,z, w = msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w
        
        # r,p,y = tf.transformations.euler_from_quaternion((x,y,z,w))





        
class tf_listen_class():
    def __init__(self,camera_frame):
        self.listener = tf.TransformListener()
        self.camera_frame = camera_frame
        now = rospy.Time.now()
        self.trans = [0,0,0]
        self.rot = [0,0,0,0]
        self.projection_matrix = np.zeros((4,4))
        self.trans_bl = [0,0,0]
        self.rot_bl = [0,0,0,0]
        self.projection_matrix_bl= np.zeros((4,4))
        self.points_in_map = []
    def wait_update_transform(self):
        t = self.listener.waitForTransform('/map',self.camera_frame,rospy.Time.now(), rospy.Duration(500.0))
        (trans, rot) = self.listener.lookupTransform('/map', self.camera_frame, rospy.Time())
        
        # update rot trans matrix
        self.trans = trans
        self.rot = rot
        self.projection_matrix = self.listener.fromTranslationRotation(trans,rot)

        t = self.listener.waitForTransform('/base_link','map',rospy.Time.now(), rospy.Duration(500.0))
        (trans_bl, rot_bl) = self.listener.lookupTransform('/base_link', 'map', rospy.Time())
        
        # update rot trans matrix
        self.trans_bl = trans_bl
        self.rot_bl = rot_bl
        self.projection_matrix_bl = self.listener.fromTranslationRotation(trans_bl,rot_bl)

       
    def transformPose(self,points):
        lane_array = LaneArray()
        lane = Lane()
        lane.header.frame_id='map'
        
        
        points_in_map = np.array(points)
        
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
                p1.pose.orientation.x,p1.pose.orientation.y,p1.pose.orientation.z,p1.pose.orientation.w = 0,0,0,0.5 #forward
            waypoint.pose = p1
            waypoint.twist.twist.linear.x =1.
            lane.waypoints.append(waypoint)

        lane_array.lanes.append(lane)
        self.points_in_map = points_in_map
        return lane_array





# =============== Node Init  ======================
rospy.init_node('vision_waypoint_generator', anonymous=True)

# =============== ROS Param Read  ======================

camera_info_topic = rospy.get_param('~camera_info_topic','/usb_camc/camera_info')
camera_frame = rospy.get_param('~camera_frame','/camerac')
pub_rate = rospy.get_param('~rate',1)
waypoints_source = rospy.get_param('~waypoints_source','main_point')
publish_topic = rospy.get_param('~publish_topic','/vision_pred_waypoints_array')
cluster_frame = rospy.get_param('~cluster_frame','map')

eps = rospy.get_param('~eps',1.0)
min_samples = rospy.get_param('~min_samples',3)
line_width = rospy.get_param('~line_width')
debug = rospy.get_param('~debug')
record_csv = rospy.get_param('~record_csv',False)
output_file = rospy.get_param('~output_file','/home/ivslab/Desktop/Waypoints_Vision_CSV/tmp.csv')

# =============== Subscribe Node Init  ======================
points_image_sub_node = Img_Sub()
main_point_sub_node = Main_Point_Sub()
camerainfo_sub_node  = CameraInfo_Sub(camera_info_topic)
current_pose_sub_node = CurrentPose_Sub()
pred_max_sub = Pred_Max_Sub()
# =============== TF Transform Node  ======================
tf_transform_camera_map = tf_listen_class(camera_frame)

# =============== Publisher Init  ======================
# posearray_pub_seg = rospy.Publisher("/lane_waypoints_array",LaneArray )
posearray_pub_seg = rospy.Publisher(publish_topic,LaneArray ,queue_size=1)
rate = rospy.Rate(pub_rate)



# =============== Visualization Setting  ======================
if debug:
    plt.ion()
    f,ax = plt.subplots(1,3,figsize=(16,4))

# =============== Record Setting  ======================
if record_csv:
    csv_file = open(output_file, 'a')


while not rospy.is_shutdown():
    
    if not (camerainfo_sub_node.camera_ok and points_image_sub_node.points_image_ok):
        print("Vision_Waypoint_Generator: camera info / Current Pose not ready")
        time.sleep(0.5)
        continue
    F_intrc_inv = camerainfo_sub_node.F_intrc_inv
    if not len(main_point_raw):
        print('Not Enough main pint raw : skipping')
        time.sleep(0.5)
        continue
    pred_max = pred_max_sub.pred_max.copy()
    


    main_point = np.array(main_point_raw).reshape(-1,2).astype(np.int32)
    distance  = points_image_sub_node.distance.copy()
    
    
    # ----------------------------------- Source Selector ----------------------------------------------
    if waypoints_source == 'main_point':
        poly_line_mask = np.zeros(distance.shape,dtype=np.uint8)
        cv2.polylines(poly_line_mask, [main_point.reshape(-1,1,2)], False, 1, line_width)
        y,x = np.nonzero(distance.copy() * poly_line_mask)
        vis_mask = poly_line_mask
    elif waypoints_source == 'main_area':
        pred_max_main = np.all(pred_max == [0,255,0], axis=-1) * 1
        y,x = np.nonzero(distance.copy() * pred_max_main)
        vis_mask = pred_max_main
    elif waypoints_source == 'alt_area':
        pred_max_alt = np.all(pred_max == [0,0,255], axis=-1) * 1
        y,x = np.nonzero(distance.copy() * pred_max_alt)
        vis_mask = pred_max_alt
    elif waypoints_source == 'line_area':
        pred_max_line = np.all(pred_max == [255,0,255], axis=-1) * 1
        y,x = np.nonzero(distance.copy() * pred_max_line)
        vis_mask = pred_max_line
    
    # --------------- Check if Point Existance in path  ---------------
    if len(x)==0:
        print('No Lidar Point In Path')
        # Publish Empty LaneArray topic ----------------
        posearray_pub_seg.publish(LaneArray())
        time.sleep(0.5)
        continue
    

    # --------------- Row based points matrix  ---------------
    # distance /100 ----> scale (m)
    points = np.stack((x*distance[y,x]/100.0,y*distance[y,x]/100.0,distance[y,x]/100.0)).T.astype(float)
    transform_points = np.ones((len(x),4),dtype=np.float32)
    transform_points[:,:3] = np.dot(F_intrc_inv,points.T).T

    tf_transform_camera_map.wait_update_transform()
    points_in_map = np.dot(tf_transform_camera_map.projection_matrix,transform_points.transpose()).transpose()
    points_in_bl = np.dot(tf_transform_camera_map.projection_matrix_bl,points_in_map.transpose()).transpose()

    if cluster_frame == 'camera':
        # Cluster in Camera frame  ---------------
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(transform_points[:,:3])
        c = db.labels_
        
        if np.all(c<1):
            print("No cluster find")
            posearray_pub_seg.publish(LaneArray())
            rate.sleep()
            continue

        clf = NearestCentroid()
        clf.fit(transform_points, c)
        centroids = clf.centroids_
        # remove -1 class 
        if -1 in c:
            centroids= np.delete(centroids,(0),axis=0)
        #sort by distance decrease order
        centroids = centroids[centroids[:,2].argsort()[::-1]]
        centroids = np.dot(tf_transform_camera_map.projection_matrix,centroids.transpose()).transpose()
    elif cluster_frame =='map':
        # Cluster in Map frame
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_in_map[:,:3])
        c = db.labels_
        if np.all(c<1):
            print("No cluster find")
            posearray_pub_seg.publish(LaneArray())
            rate.sleep()
            continue
        clf = NearestCentroid()
        clf.fit(points_in_map, c)
        centroids = clf.centroids_
        # remove -1 class 
        if -1 in c:
            centroids= np.delete(centroids,(0),axis=0)
        #sort by distance decrease order
        centroids = centroids[centroids[:,2].argsort()[::-1]]
    elif cluster_frame =='none':
        centroids = points_in_map

    centroids_in_bl = np.dot(tf_transform_camera_map.projection_matrix_bl,centroids.transpose()).transpose()
    posearray= tf_transform_camera_map.transformPose(centroids[::-1])
    posearray_pub_seg.publish(posearray)
    

    # Vis debug -----------------------------------------------------
    pred_max_line = np.all(pred_max == [255,0,255], axis=-1) * 1
    yy,xx = np.nonzero(distance.copy() * pred_max_line)
    if len(xx)!=0:
        # --------------- Row based points matrix  ---------------
        # distance /100 ----> scale (m)
        line_points = np.stack((xx*distance[yy,xx]/100.0,yy*distance[yy,xx]/100.0,distance[yy,xx]/100.0)).T.astype(float)
        transform_line_points = np.ones((len(xx),4),dtype=np.float32)
        transform_line_points[:,:3] = np.dot(F_intrc_inv,line_points.T).T
        line_in_map = np.dot(tf_transform_camera_map.projection_matrix,transform_line_points.transpose()).transpose()
        line_in_bl = np.dot(tf_transform_camera_map.projection_matrix_bl,line_in_map.transpose()).transpose()


    

    

    # Debug Visualization
    if debug :
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[0].imshow(vis_mask)
        ax[0].scatter(x,y,s=1,c=c,cmap=plt.cm.cool)
        ax[0].set_title("Cluster {}".format(len(centroids)))
        ax[1].imshow(vis_mask)
        py,px = np.nonzero(distance)
        # ax[1].scatter(px,py,s=1,c=distance[py,px],cmap=plt.cm.cool)
        # ax[1].set_title("Planning Result Overlay with lidar")
        ax[2].scatter(-1*points_in_bl[:,1],points_in_bl[:,0],c='b')
        ax[2].scatter(-1*line_in_bl[:,1],line_in_bl[:,0],c="g")
        ax[2].scatter(-1*centroids_in_bl[:,1],centroids_in_bl[:,0],c="r")
        ax[2].set_ylim(0, 50)
        ax[2].set_xlim(-25, 25)

        f.canvas.draw()
    # if plot_base_link:
        
    if record_csv:
        points_in_map = tf_transform_camera_map.points_in_map
        dataset = pd.DataFrame({'X': points_in_map[:, 0], 'Y': points_in_map[:, 1]})
        print(dataset)
        dataset.to_csv(csv_file, header=csv_file.tell()==0)
    rate.sleep()
