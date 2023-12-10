import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la


def wrap_angle(theta):
    if theta < -np.pi:
        wrapped_angle = 2 * np.pi + theta
    elif theta > np.pi:
        wrapped_angle = theta - 2 * np.pi
    else:
        wrapped_angle = theta

    return wrapped_angle



def compute_angle(point_0, point_1, point_2):
    v_1 = point_1 - point_0
    v_2 = point_2 - point_0

    dot = v_1.dot(v_2)
    det = v_1[0] * v_2[1] - v_1[1] * v_2[0]
    theta = np.arctan2(det, dot)

    return theta



class Track:
    def __init__(self, initial_x=0, initial_y=0, initial_psi=0.0):
        self.segments = []
        self.cumulative_len = 0.0
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.initial_psi = initial_psi
        self.track_width = 1.0
        self.slack = 0.1
        self.track_length = 0.0


    def add_segment(self, curvature, segment_length):
        self.segments.append([curvature, segment_length])

    def interpolate_straight_line(self, x1, y1, x2, y2, num_points=10):
        return np.linspace(x1, x2, num_points), np.linspace(y1, y2, num_points)

    def interpolate_arc(self, x_center, y_center, radius, start_angle, end_angle, num_points=10):
        angles = np.linspace(start_angle, end_angle, num_points)
        x_values = x_center + radius * np.sin(angles)
        y_values = y_center - radius * np.cos(angles)
        return x_values, y_values

    def get_keypts(self):
        x, y, psi = self.initial_x, self.initial_y, self.initial_psi
        x_values, y_values = [], []
        self.key_pts = []
        for curvature, segment_length in self.segments:
            # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
            self.cumulative_len += segment_length
            end_x, end_y, end_psi, x_seg, y_seg = self.next_position_orientation(x, y, psi, curvature, segment_length)            
            kpt = [end_x, end_y, end_psi, self.cumulative_len, segment_length,curvature].copy()
            self.key_pts.append(kpt.copy())
            x = end_x
            y = end_y
            psi = end_psi
            
            
            x_values.extend(x_seg)
            y_values.extend(y_seg)
        print(1)
        self.track_length = self.cumulative_len
        self.key_pts = np.stack(self.key_pts)
        print(self.key_pts)
        
    def next_position_orientation(self, x, y, psi, curvature, segment_length):
        if curvature == 0:  # Straight line
            x_next = x + segment_length * np.cos(psi)
            y_next = y + segment_length * np.sin(psi)
            psi_next = psi
            x_points, y_points = self.interpolate_straight_line(x, y, x_next, y_next)
        else:  # Circular path
            radius = 1 / curvature
            angle_change = segment_length / radius
            psi_next = psi + angle_change
            x_center = x - radius * np.sin(psi)
            y_center = y + radius * np.cos(psi)
            x_next = x_center + radius * np.sin(psi_next)
            y_next = y_center - radius * np.cos(psi_next)
            x_points, y_points = self.interpolate_arc(x_center, y_center, radius, psi, psi_next)

        return x_next, y_next, psi_next, x_points, y_points

   
    def compute_angle(self,point_0, point_1, point_2):
        v_1 = point_1 - point_0
        v_2 = point_2 - point_0

        dot = v_1.dot(v_2)
        det = v_1[0] * v_2[1] - v_1[1] * v_2[0]
        theta = np.arctan2(det, dot)

        return theta



    
    # def global_to_local(self, xy_coord, line='center'):
    #     if self.key_pts is None:
    #         raise ValueError('Track key points have not been defined')

    #     x = xy_coord[0]
    #     y = xy_coord[1]
    #     psi = xy_coord[2]

    #     pos_cur = np.array([x, y])
    #     cl_coord = None

    #     for i in range(1, self.key_pts.shape[0]):
    #         # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
    #         x_s = self.key_pts[i - 1, 0]
    #         y_s = self.key_pts[i - 1, 1]
    #         psi_s = self.key_pts[i - 1, 2]
    #         curve_s = self.key_pts[i - 1, 5]
    #         x_f = self.key_pts[i, 0]
    #         y_f = self.key_pts[i, 1]
    #         psi_f = self.key_pts[i, 2]
    #         curve_f = self.key_pts[i, 5]

    #         l = self.key_pts[i, 4]

    #         pos_s = np.array([x_s, y_s])
    #         pos_f = np.array([x_f, y_f])

    #         # Check if at any of the segment start or end points
    #         if la.norm(pos_s - pos_cur) == 0:
    #             # At start of segment
    #             s = self.key_pts[i - 1, 3]
    #             e_y = 0
    #             e_psi = np.unwrap([psi_s, psi])[1] - psi_s
    #             cl_coord = (s, e_y, e_psi)                
    #             break
    #         if la.norm(pos_f - pos_cur) == 0:
    #             # At end of segment
    #             s = self.key_pts[i, 3]
    #             e_y = 0
    #             e_psi = np.unwrap([psi_f, psi])[1] - psi_f
    #             cl_coord = (s, e_y, e_psi)                
    #             break

    #         if curve_f == 0:                
    #             # Check if on straight segment
    #             if np.abs(self.compute_angle(pos_s, pos_cur, pos_f)) <= np.pi / 2 and np.abs(
    #                     self.compute_angle(pos_f, pos_cur, pos_s)) <= np.pi / 2:
    #                 v = pos_cur - pos_s
    #                 ang = self.compute_angle(pos_s, pos_f, pos_cur)
    #                 e_y = la.norm(v) * np.sin(ang)
    #                 # Check if deviation from centerline is within track width plus some slack for current segment
    #                 # (allows for points outside of track boundaries)
                  
                    
    #                 if np.abs(e_y) <= self.track_width / 2 + self.slack:
    #                     d = la.norm(v) * np.cos(ang)
    #                     s = self.key_pts[i - 1, 3] + d
    #                     e_psi = np.unwrap([psi_s, psi])[1] - psi_s
    #                     cl_coord = (s, e_y, e_psi)                        
    #                     break
    #                 else:                        
    #                     continue
    #             else:
    #                 continue                
    #         else:
    #             # Check if on curved segment                                
    #             r = 1 / curve_f
    #             dir = np.sign(r)

    #             # Find coordinates for center of curved segment
    #             x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
    #             y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
    #             curve_center = np.array([x_c, y_c])

    #             span_ang = l / r
    #             cur_ang = self.compute_angle(curve_center, pos_s, pos_cur)
    #             if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
    #                 v = pos_cur - curve_center
    #                 e_y = -np.sign(dir) * (la.norm(v) - np.abs(r))
    #                 # Check if deviation from centerline is within track width plus some slack for current segment
    #                 # (allows for points outside of track boundaries)
    #                 if np.abs(e_y) <= self.track_width / 2 + self.slack:
    #                     d = np.abs(cur_ang) * np.abs(r)
    #                     s = self.key_pts[i - 1, 3] + d
    #                     e_psi = np.unwrap([psi_s + cur_ang, psi])[1] - (psi_s + cur_ang)
    #                     cl_coord = (s, e_y, e_psi)                        
    #                     break
    #                 else:                        
    #                     continue
    #             else:                    
    #                 continue

    #     if line == 'inside':
    #         cl_coord = (cl_coord[0], cl_coord[1] - self.track_width / 5, cl_coord[2])
    #     elif line == 'outside':
    #         cl_coord = (cl_coord[0], cl_coord[1] + self.track_width / 5, cl_coord[2])
    #     elif line == 'pid_offset':
    #         # PID controller tends to cut to the inside of the track
    #         cl_coord = (cl_coord[0], cl_coord[1] + (0.1 * self.track_width / 2), cl_coord[2])

    #     # if cl_coord is None:
    #     #     raise ValueError('Point is out of the track!')

    #     return cl_coord

    def global_to_local(self, xy_coord, line='center'):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        x = xy_coord[0]
        y = xy_coord[1]
        psi = xy_coord[2]

        pos_cur = np.array([x, y])
        cl_coord = None

        for i in range(1, self.key_pts.shape[0]):
            # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
            x_s = self.key_pts[i - 1, 0]
            y_s = self.key_pts[i - 1, 1]
            psi_s = self.key_pts[i - 1, 2]
            curve_s = self.key_pts[i - 1, 5]
            x_f = self.key_pts[i, 0]
            y_f = self.key_pts[i, 1]
            psi_f = self.key_pts[i, 2]
            curve_f = self.key_pts[i, 5]

            l = self.key_pts[i, 4]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            # Check if at any of the segment start or end points
            if la.norm(pos_s - pos_cur) == 0:
                # At start of segment
                s = self.key_pts[i - 1, 3]
                e_y = 0
                e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                cl_coord = (s, e_y, e_psi)                
                break
            if la.norm(pos_f - pos_cur) == 0:
                # At end of segment
                s = self.key_pts[i, 3]
                e_y = 0
                e_psi = np.unwrap([psi_f, psi])[1] - psi_f
                cl_coord = (s, e_y, e_psi)                
                break
                
            if curve_f == 0: 
                           
                # Check if on straight segment
                if np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi / 2 and np.abs(
                        compute_angle(pos_f, pos_cur, pos_s)) <= np.pi / 2:
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    e_y = la.norm(v) * np.sin(ang)
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                  
                    
                    if np.abs(e_y) <= self.track_width / 2 + self.slack:
                        d = la.norm(v) * np.cos(ang)
                        s = self.key_pts[i - 1, 3] + d
                        e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                        cl_coord = (s, e_y, e_psi)                        
                        break
                    else:                        
                        continue
                else:
                    continue                
            else:
                # Check if on curved segment                                
                print("@@@@@@@")
                print(curve_f)  
                r = 1 / curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
                y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
                curve_center = np.array([x_c, y_c])

                span_ang = l / r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)
                if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
                    v = pos_cur - curve_center
                    e_y = -np.sign(dir) * (la.norm(v) - np.abs(r))
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    if np.abs(e_y) <= self.track_width / 2 + self.slack:
                        d = np.abs(cur_ang) * np.abs(r)
                        s = self.key_pts[i - 1, 3] + d
                        e_psi = np.unwrap([psi_s + cur_ang, psi])[1] - (psi_s + cur_ang)
                        cl_coord = (s, e_y, e_psi)                        
                        break
                    else:                        
                        continue
                else:                    
                    continue

        if line == 'inside':
            cl_coord = (cl_coord[0], cl_coord[1] - self.track_width / 5, cl_coord[2])
        elif line == 'outside':
            cl_coord = (cl_coord[0], cl_coord[1] + self.track_width / 5, cl_coord[2])
        elif line == 'pid_offset':
            # PID controller tends to cut to the inside of the track
            cl_coord = (cl_coord[0], cl_coord[1] + (0.1 * self.track_width / 2), cl_coord[2])

        # if cl_coord is None:
        #     raise ValueError('Point is out of the track!')

        return cl_coord
    

    def local_to_global(self, cl_coord):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        # s = np.mod(cl_coord[0], self.track_length) # Distance along current lap
        s = cl_coord[0]
        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length

        e_y = cl_coord[1]
        e_psi = cl_coord[2]

        # Find key point indicies corresponding to current segment
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        key_pt_idx_s = np.where(s >= self.key_pts[:, 3])[0][-1]
        key_pt_idx_f = key_pt_idx_s + 1
        seg_idx = key_pt_idx_s
        print("key_pt_idx_s = "  + str(key_pt_idx_s))

        x_s = self.key_pts[key_pt_idx_s, 0]
        y_s = self.key_pts[key_pt_idx_s, 1]
        psi_s = self.key_pts[key_pt_idx_s, 2]
        curve_s = self.key_pts[key_pt_idx_s, 5]
        x_f = self.key_pts[key_pt_idx_f, 0]
        y_f = self.key_pts[key_pt_idx_f, 1]
        psi_f = self.key_pts[key_pt_idx_f, 2]
        curve_f = self.key_pts[key_pt_idx_f, 5]

        l = self.key_pts[key_pt_idx_f, 4]
        d = s - self.key_pts[key_pt_idx_s, 3]  # Distance along current segment

        if curve_f == 0:
            # Segment is a straight line
            x = x_s + (x_f - x_s) * d / l + e_y * np.cos(psi_f + np.pi / 2)
            y = y_s + (y_f - y_s) * d / l + e_y * np.sin(psi_f + np.pi / 2)
            psi = wrap_angle(psi_f + e_psi)
        else:
            r = 1 / curve_f
            dir = np.sign(r)

            # Find coordinates for center of curved segment
            x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
            y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)

            # Angle spanned up to current location along segment
            span_ang = d / np.abs(r)

            # Angle of the tangent vector at the current location
            psi_d = wrap_angle(psi_s + dir * span_ang)

            ang_norm = wrap_angle(psi_s + dir * np.pi / 2)
            ang = -np.sign(ang_norm) * (np.pi - np.abs(ang_norm))

            x = x_c + (np.abs(r) - dir * e_y) * np.cos(ang + dir * span_ang)
            y = y_c + (np.abs(r) - dir * e_y) * np.sin(ang + dir * span_ang)
            psi = wrap_angle(psi_d + e_psi)
        return (x, y, psi)

        
    def plot(self):
        
        x, y, psi = self.initial_x, self.initial_y, self.initial_psi
        x_values, y_values = [], []
        
        for curvature, segment_length in self.segments:
            x, y, psi, x_seg, y_seg = self.next_position_orientation(x, y, psi, curvature, segment_length)
            x_values.extend(x_seg)
            y_values.extend(y_seg)

        plt.plot(x_values, y_values, marker='o')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Racing Line with Interpolated Points')
        plt.grid(True)
        plt.show()

# Example usage
track = Track()
track.add_segment(0.0, 0.0)
track.add_segment(0.3, 1)
track.add_segment(-0.3, 2)
track.add_segment(0.0, 3)
track.add_segment(0.0, 3)
track.get_keypts()
pose = [2.1, -0.39, 0.0]
a = track.global_to_local(pose)
b = track.local_to_global(a)
print(a)
print(a[2]*180/np.pi)
print(b)
track.plot()
