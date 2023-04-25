import math
import numpy as np
import matplotlib.pyplot as plt
import bisect
from utils.pycubicspline import Spline, Spline2D
from utils.projection import Projection

class Track():
    def __init__(self,filename):
        center = np.loadtxt(filename, delimiter=",", dtype = float)
        self.x_center = center[:,0]
        self.y_center = center[:,1]
        self.track_width = 1
        if center.shape[1] > 2:
            self.track_width_in = center[:,2]
            self.track_width_out = center[:,3]
        
        self.center_line = np.transpose(center[:,0:2])


        self.load_raceline()
        self._calc_theta_track()
        self._calc_track_length()

        self.psi_init = np.arctan2(self.y_raceline[1] - self.y_raceline[0], self.x_raceline[1] - self.x_raceline[0])
        self.x_init = self.x_raceline[0]
        self.y_init = self.y_raceline[0]
        self.vx_init = 0
    
    def _calc_theta_track(self):
        """	calculate theta for the center line
        """
        diff = np.diff(self.center_line)
        theta_track = np.cumsum(np.linalg.norm(diff, 2, axis=0))
        self.theta_track = np.concatenate([np.array([0]), theta_track])
    
    def _calc_track_length(self):
        """	calculate track length using (x,y) for center line
        """
        center = self.center_line
        # connect first and last point
        center = np.concatenate([center, center[:,0].reshape(-1,1)], axis=1) 
        diff = np.diff(center)
        self.track_length = np.sum(np.linalg.norm(diff, 2, axis=0))


    def load_raceline(self):
	    n_samples = 2*self.x_center.shape[0]-1
	    self._load_raceline(
			wx=self.x_center,
			wy=self.y_center,
			n_samples=n_samples
			)

    def _load_raceline(self, wx, wy, n_samples, v=None, t=None):
        self.spline = Spline2D(wx, wy)
        x, y = wx, wy
        theta = self.spline.s
        
        self.x_raceline = np.array(x)
        self.y_raceline = np.array(y)
        self.raceline = np.array([x, y])
        
        if v is not None:
            self.v_raceline = v
            self.t_raceline = t
            self.spline_v = Spline(theta, v)
    
    def _param2xy(self, theta):
        """	finds (x,y) coordinate on center line for a given theta
        """
        theta_track = self.theta_track
        idt = 0
        while idt<theta_track.shape[0]-1 and theta_track[idt]<=theta:
            idt+=1
        deltatheta = (theta-theta_track[idt-1])/(theta_track[idt]-theta_track[idt-1])
        x = self.x_center[idt-1] + deltatheta*(self.x_center[idt]-self.x_center[idt-1])
        y = self.y_center[idt-1] + deltatheta*(self.y_center[idt]-self.y_center[idt-1])
        return x, y
    
    def _xy2param(self, x, y):
        """	finds theta on center line for a given (x,y) coordinate
        """
        center_line = self.center_line
        theta_track = self.theta_track

        optxy, optidx = self.project(x, y, center_line)
        distxy = np.linalg.norm(optxy-center_line[:,optidx],2)
        dist = np.linalg.norm(center_line[:,optidx+1]-center_line[:,optidx],2)
        deltaxy = distxy/dist
        if optidx==-1:
            theta = theta_track[optidx] + deltaxy*(self.track_length-theta_track[optidx])
        else:
            theta = theta_track[optidx] + deltaxy*(theta_track[optidx+1]-theta_track[optidx])
        theta = theta % self.track_length
        return theta


    def project(self, x, y, raceline):
        """	finds projection for (x,y) on a raceline
        """
        point = [(x, y)]
        n_waypoints = raceline.shape[1]

        proj = np.empty([2,n_waypoints])
        dist = np.empty([n_waypoints])
        for idl in range(-1, n_waypoints-1):
            line = [raceline[:,idl], raceline[:,idl+1]]
            proj[:,idl], dist[idl] = Projection(point, line)
        optidx = np.argmin(dist)
        if optidx == n_waypoints-1:
            optidx = -1
        optxy = proj[:,optidx]
        return optxy, optidx



class Wall():
    def __init__(self,filename):
        center = np.loadtxt(filename, delimiter=",", dtype = float)
        self.x_center = center[:,0]
        self.y_center = center[:,1]
        self.track_width = 1
        
        self.center_line = np.transpose(center[:,0:2])


        self.load_raceline()
        self._calc_theta_track()
        self._calc_track_length()

        self.psi_init = np.arctan2(self.y_raceline[1] - self.y_raceline[0], self.x_raceline[1] - self.x_raceline[0])
        self.x_init = self.x_raceline[0]
        self.y_init = self.y_raceline[0]
        self.vx_init = 0
    
    def _calc_theta_track(self):
        """	calculate theta for the center line
        """
        diff = np.diff(self.center_line)
        theta_track = np.cumsum(np.linalg.norm(diff, 2, axis=0))
        self.theta_track = np.concatenate([np.array([0]), theta_track])
    
    def _calc_track_length(self):
        """	calculate track length using (x,y) for center line
        """
        center = self.center_line
        # connect first and last point
        center = np.concatenate([center, center[:,0].reshape(-1,1)], axis=1) 
        diff = np.diff(center)
        self.track_length = np.sum(np.linalg.norm(diff, 2, axis=0))


    def load_raceline(self):
	    n_samples = 2*self.x_center.shape[0]-1
	    self._load_raceline(
			wx=self.x_center,
			wy=self.y_center,
			n_samples=n_samples
			)

    def _load_raceline(self, wx, wy, n_samples, v=None, t=None):
        self.spline = Spline2D(wx, wy)
        x, y = wx, wy
        theta = self.spline.s
        
        self.x_raceline = np.array(x)
        self.y_raceline = np.array(y)
        self.raceline = np.array([x, y])
        
        if v is not None:
            self.v_raceline = v
            self.t_raceline = t
            self.spline_v = Spline(theta, v)
    
    def _param2xy(self, theta):
        """	finds (x,y) coordinate on center line for a given theta
        """
        theta_track = self.theta_track
        idt = 0
        while idt<theta_track.shape[0]-1 and theta_track[idt]<=theta:
            idt+=1
        deltatheta = (theta-theta_track[idt-1])/(theta_track[idt]-theta_track[idt-1])
        x = self.x_center[idt-1] + deltatheta*(self.x_center[idt]-self.x_center[idt-1])
        y = self.y_center[idt-1] + deltatheta*(self.y_center[idt]-self.y_center[idt-1])
        return x, y
    
    def _xy2param(self, x, y):
        """	finds theta on center line for a given (x,y) coordinate
        """
        center_line = self.center_line
        theta_track = self.theta_track

        optxy, optidx = self.project(x, y, center_line)
        distxy = np.linalg.norm(optxy-center_line[:,optidx],2)
        dist = np.linalg.norm(center_line[:,optidx+1]-center_line[:,optidx],2)
        deltaxy = distxy/dist
        if optidx==-1:
            theta = theta_track[optidx] + deltaxy*(self.track_length-theta_track[optidx])
        else:
            theta = theta_track[optidx] + deltaxy*(theta_track[optidx+1]-theta_track[optidx])
        theta = theta % self.track_length
        return theta


    def project(self, x, y, raceline):
        """	finds projection for (x,y) on a raceline
        """
        point = [(x, y)]
        n_waypoints = raceline.shape[1]

        proj = np.empty([2,n_waypoints])
        dist = np.empty([n_waypoints])
        for idl in range(-1, n_waypoints-1):
            line = [raceline[:,idl], raceline[:,idl+1]]
            proj[:,idl], dist[idl] = Projection(point, line)
        optidx = np.argmin(dist)
        if optidx == n_waypoints-1:
            optidx = -1
        optxy = proj[:,optidx]
        return optxy, optidx
