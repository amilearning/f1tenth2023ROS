
* How to run
$ python3 pub_track.py
    - rosnode name : track_info_node
    - rostopic name: track_info
    - message structure:
        data[].pose.position.x = x 
        data[].pose.position.y = y
        data[].pose.position.z = velocity
        data[].pose.orientation.x = curvature
        data[].pose.orientation.y = left wall width 
        data[].pose.orientation.z = right wall width 

* Information
    - track_extraction: track extraction from pgm file 
    - track_fitting: track smoothing from cvs file(extracted track)
    - result: txt files of smoothed track.
