## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

The PR2 Robot uses an RGB-D camera to obtain the scene data. The camera picks up the color and the depth of the objects in it's view, and this is represented by a point cloud. In order to complete this project, we need to filter, cluster, and identify the objects in the scene. These processes are included in the `pcl_callback()` function. 

Often times the point cloud contains a lot of excess information in the point cloud that can interfere with achieving our end goal, to locate the objects on the table and find their position. Excess information includes other objects in the scene and noise. Luckily, there are several filtering techniques we can use to find exactly what we're looking for. Here's a list of filters and description I made and used in this project:

1. **Voxel Downsampling Filter**: RGB-D cameras often output very dense point clouds, and usually it's advantageous to downsample the data. Using a volume element (voxel) downsampling filter, we can create cells of a certain size (leaf size) and average the RGB-D values of all the Points in a particular cell can be averaged and output as one single point with RGB-D values. I chose the grid to be [0.003 m x 0.003 mx 0.003 m] (leaf = 0.003). The `pcl` library has a built in function `make_voxel_grid_filter()` to perform voxel downsampling:

```sh
    vox = cloud.make_voxel_grid_filter() #cloud is raw pcl
    LEAF_SIZE = 0.003
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter() #filtered pcl

```

2. **Pass Through Filter**: The pass through filter is essentially a method to crop the region of interest. We can figure out the dimensions of the table and along what axes it sits, and then we can crop out the rest of the image. This allows us to focus on just the table, as this is our region of interest. For this project, I implemented two separate pass through filters, one along the y-axis and one along the z-axis. We have to set a range along the axes for which points we want to include.

```sh
  #Passthrough filter along z-axis
    passThroughZ = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passThroughZ.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passThroughZ.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passThroughZ.filter()

  #Passthrough filter along y-axis
    passThroughY = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passThroughY.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passThroughY.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passThroughY.filter()
```
3. **RANSAC Filter**: Random Sample Consensus (RANSAC) is an algorithm used to identify points in the dataset that belong in a particular model. In this particular case, I model the table top as a plane, and filter it out of the latest `cloud_filtered` point cloud. After applying the previous two filters, the latest point cloud should contain only the tabletop and the objects on it. Therefore, by applying this RANSAC filter, we can determine the points that don't fall into the filter criteria are the objects. Here's the code of how I implemented the RANSAC filter:

```sh
    seg = cloud_filtered.make_segmenter() #Create segmentation object
    seg.set_model_type(pcl.SACMODEL_PLANE) #Model it to a plane
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01	#max distance for point to fit criteria
    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment()  #call segment function

    pcl_table_cloud = cloud_filtered.extract(inliers, negative=False) #inliers - table cloud
    pcl_object_cloud = cloud_filtered.extract(inliers, negative=True) #outliers - object cloud
```

4. **Outlier Removal Filter**: Now that we have separated the objects from the table, we can apply one last filter to reduce the noise in the point cloud and increase the clustering time and accuracy. We assume a Gaussian distribution when determining which points should be excluded. Here's my implementation of the outlier removal filter: 

```sh
   outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    x = 1.0

    # Any point with a mean distance larger than mean distance+x*std_dev will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()

```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

Now that we have filtered the original point cloud to just the objects, we can cluster separate objects by looking for spots with a high density of points. This is called clustering and there are several clustering algorithms and methods. I implement the Euclidean clustering technique as there are built in functions in the `pcl` library. First, we create a k-d tree from the object-cloud. This k-d tree helps us cluster the points based on the distance to neighbors and helps us create an accurate representation of the individual objects. This algorithm uses only the distance between points and not color, therefore the input is simply the XYZ cloud. We can set the cluster tolerance and minimum and maximum cluster sizes to help the function determine the clusters. Here's my implementation of clustering for segmentation:

```sh
    white_cloud = XYZRGB_to_XYZ(pcl_object_cloud)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.005)
    ec.set_MinClusterSize(150)
    ec.set_MaxClusterSize(50000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
```

Now, we can color each of these clusters to help us visualize it in RVIZ.

```sh
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
             color_cluster_point_list.append([white_cloud[indice][0],
                                              white_cloud[indice][1],
                                              white_cloud[indice][2],
                                     rgb_to_float(cluster_color[j])])


    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
``` 

Now we have filtered and clustered all the objects in the scene. Next, we need to implement our object recognition techniques into the `pcl_callback()` function to identify the separate objects.

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Before we get to identifying the objects in the scene, we need to train a support vector machine (SVM) to identify all the objects in the scenes. We can do this by creating histograms of the colors and surface normals for each object in various orientations. We can build color and surface normal histograms in the training environment provided in Exercise 3. I changed the launch file to open objects used in the project. Each object will have a unique histogram and we can feed this data into the SVM to help identify objects in our RVIZ environment.

1. **Color**: To train the SVM to determine the color of each object. The code below is found in the `features.py` in the sensor_stick package. I implemented my color histograms to use the HSV color space, and we can do this simply by running our RGB points through the `rgb_to_hsv` function also in `features.py`. HSV makes it easier to identify the colors through all brightness levels. Once we have the color histogram, we can normalize it so it can be used to be compared to different sets of data.

```sh

def compute_color_histograms(cloud, using_hsv=True):
    point_colors_list = []
    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
 # Compute histograms
    ch1_hist = np.histogram(channel_1_vals, bins = 32, range = (0,256))
    ch2_hist = np.histogram(channel_2_vals, bins = 32, range = (0,256))
    ch3_hist = np.histogram(channel_3_vals, bins = 32, range = (0,256))

 # Concatenate and normalize the histograms
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0])).astype(np.float64)
    normed_features = hist_features/np.sum(hist_features) #Normalize the results
    return normed_features 
```

**RGB to HSV function**
```sh
def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

```

2. **Surface Normals**: Now that we have modeled a color histogram for all the objects, we can apply a similar technique to further improve the prediction accuracy. Instead of using color, we use the shape of the object and determine the surface normals for each point in the cloud. Each object will produce a unique range of surface normals, which can be visualized in yet another histogram. We can then normalize this surface normal histogram as well. Below is my implementation of the code, found in the `features.py` of the sensor_stick package:

```sh
def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    #Compute histograms
    chx_hist = np.histogram(norm_x_vals, bins = 32, range = (0,256))
    chy_hist = np.histogram(norm_y_vals, bins = 32, range = (0,256))
    chz_hist = np.histogram(norm_z_vals, bins = 32, range = (0,256))
    #Concatenate and normalize the histograms
    hist_features = np.concatenate((chx_hist[0], chy_hist[0], chz_hist[0])).astype(np.float64)
    normed_features = hist_features/np.sum(hist_features) #normalize histogram
    return normed_features
``` 

In order to obtain the color and surface normal histograms, we need to run `training.launch` and `capture_features.py` from the sensor_stick package. Make sure that the `capture_features.py` contains the objects from the project and not the demo. I scan each object 15 times in different orientations. The improvement in accuracy can be seen in the histogram provided below. We can concatenate the two normalized histograms to produce a single histogram that considers both color and surface normals, as such: 

```sh
    chists = compute_color_histograms(sample_cloud, using_hsv=True)
    normals = get_normals(sample_cloud)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))
    labeled_features.append([feature, model_name])
```
Running `capture_features.py` outputs un-normalized and normalized matrices, as well as a `training_set.sav` file. This file is used to train the SVM by running the command `rosrun sensor_stick train_svm.py`. `train_svm.py` outputs a file called `model.sav`, which essentially allows us to predict an object given its RGB-D point cloud. 

The accuracy of my SVM model is 100%, and this is likely because I scanned each object 15 times.

[histogram]

Here's how I implemented object recognition into my project:

```sh
    det_obj_label = []
    det_obj = []

    for index, pts_list in enumerate(cluster_indices):
	pcl_cluster = pcl_object_cloud.extract(pts_list)
	ros_cluster = pcl_to_ros(pcl_cluster)

	color_hist = compute_color_histograms(ros_cluster, using_hsv=True)
	
	norms = get_normals(ros_cluster)
	norm_hist = compute_normal_histograms(norms)

	features = np.concatenate((color_hist, norm_hist))
	predict = clf.predict(scaler.transform(features.reshape(1,-1)))
	label = encoder.inverse_transform(predict)[0]
	det_obj_label.append(label)

```

We can publish the labels of the object in RVIZ:

```sh
	label_pos = list(white_cloud[pts_list[0]])
	label_pos[2] += .2
	object_markers_pub.publish(make_label(label,label_pos,index))

	do = DetectedObject()
	do.label = label
	do.cloud = ros_cluster
	det_obj.append(do)
    rospy.loginfo('Detected {} objects: {}'.format(len(det_obj_label), det_obj_label))
    detected_objects_pub.publish(det_obj)
```

 ![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further. 





