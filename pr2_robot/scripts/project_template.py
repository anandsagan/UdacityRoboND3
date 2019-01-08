#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    cloud = ros_to_pcl(pcl_msg)

    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.003

    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

### Statistical Outlier Removal Filter
    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()


### PassThrough Filter
    # Create a PassThrough filter object first across the Z axis
    passThroughZ = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passThroughZ.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passThroughZ.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passThroughZ.filter()

## Now, Create a PassThrough filter object across the Y axis
    passThroughY = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passThroughY.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passThroughY.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passThroughY.filter()


### RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()


    ### Extract inliers and outliers
    pcl_table_cloud = cloud_filtered.extract(inliers, negative=False)
    pcl_object_cloud = cloud_filtered.extract(inliers, negative=True)

### Euclidean Clustering
    # Go from XYZRGB to RGB since to build the k-d tree we only needs spatial data
    white_cloud = XYZRGB_to_XYZ(pcl_object_cloud)
    # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.005)
    ec.set_MinClusterSize(150)
    ec.set_MaxClusterSize(50000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    #Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
             color_cluster_point_list.append([white_cloud[indice][0],
                                              white_cloud[indice][1],
                                              white_cloud[indice][2],
                                     rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
 
    #Convert PCL data to ROS messages
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_object_cloud = pcl_to_ros(pcl_object_cloud)
    ros_table_cloud = pcl_to_ros(pcl_table_cloud)

    #Publish ROS messages
    pcl_objects_pub.publish(ros_object_cloud)
    pcl_table_pub.publish(ros_table_cloud)
    pcl_cluster_pub.publish(ros_cluster_cloud)
    

    det_obj_label = []
    det_obj = []
        # Grab the points for the cluster
    for index, pts_list in enumerate(cluster_indices):
	pcl_cluster = pcl_object_cloud.extract(pts_list)
	ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute associated feature vector
	color_hist = compute_color_histograms(ros_cluster, using_hsv=True)
	
	norms = get_normals(ros_cluster)
	norm_hist = compute_normal_histograms(norms)

        # Make prediction
	features = np.concatenate((color_hist, norm_hist))
	predict = clf.predict(scaler.transform(features.reshape(1,-1)))
	label = encoder.inverse_transform(predict)[0]
	det_obj_label.append(label)

        # Publish a label into RViz
	label_pos = list(white_cloud[pts_list[0]])
	label_pos[2] += .2
	object_markers_pub.publish(make_label(label,label_pos,index))

	do = DetectedObject()
	do.label = label
	do.cloud = ros_cluster
	det_obj.append(do)

    
    rospy.loginfo('Detected {} objects: {}'.format(len(det_obj_label), det_obj_label))

    # Publish the list of detected objects
    detected_objects_pub.publish(det_obj)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    true_object_list = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    true_list = []
    
    for i in range(len(true_object_list)):
	true_list.append(true_object_list[i]['name'])

    print "\n"  
    print "Need to find: "
    print true_list
    print "\n"
    trueset = set(true_list)
    detset = set(det_obj_label)


    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    if detset == trueset:
	try:
            pr2_mover(det_obj)
	except rospy.ROSInterruptException:
            pass
    else:
	rospy.loginfo("Wrong object(s) detected")

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    #Initialize variables
    test_scene_num = Int32()
    object_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    arm_name = String()
    yaml_dict_list = []
    test_scene_num.data = 2
    
    dropbox_pose = []
    dropbox_name = []
    dropbox_group = []

    true_object_list = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    for i in range(0, len(dropbox_param)):
	dropbox_pose.append(dropbox_param[i]['position'])
	dropbox_group.append(dropbox_param[i]['group'])

    centroids = [] # to be list of tuples (x, y, z)
    labels = []

    for i in range(0, len(true_object_list)):
        object_name.data = true_object_list[i]['name' ]
        object_group     = true_object_list[i]['group']
	for j in object_list:
	    labels.append(j.label)
	    point_arr = ros_to_pcl(j.cloud).to_array()
	    avg = np.mean(point_arr, axis=0)[:3]
	    centroids.append(avg)

	if object_group == 'red':
	    arm_name.data = 'left'
	    place_pose.position.x = dropbox_pose[0][0] - random.randint(.05,.1)
	    place_pose.position.y = dropbox_pose[0][1]
	    place_pose.position.z = dropbox_pose[0][2]
	else:
	    arm_name.data = 'right'
	    place_pose.position.x = dropbox_pose[1][0] - random.randint(.05,.1)
	    place_pose.position.y = dropbox_pose[1][1]
	    place_pose.position.z = dropbox_pose[1][2]
        try:
            index = labels.index(object_name.data)
            pick_pose.position.x = np.asscalar(centroids[index][0])
            pick_pose.position.y = np.asscalar(centroids[index][1])
            pick_pose.position.z = np.asscalar(centroids[index][2])
        except ValueError:
            continue


        # Create a list of dictionaries for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            #Insert your message variables to be sent as a service requestc
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    #Output your request parameters into output yaml file
    yaml_filename = 'output_'+str(test_scene_num.data)+'.yaml'
    send_to_yaml(yaml_filename, yaml_dict_list)

    return


if __name__ == '__main__':

    
    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback,queue_size=1)
    
    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2,queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pr2_base_mover_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10)

    #Load Model From disk
    model = pickle.load(open('model.sav','rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
	rospy.spin()
