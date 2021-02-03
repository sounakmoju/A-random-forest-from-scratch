import sys
import numpy as np


group_dis_tolerance=0.1
class point_Grouper(object):
    def group_points(self,points):
        group_assignment=[]
        groups=[]
        group_index=0
        for point in points:
            nearest_group_index=self._determine_nearest_group(point,groups)
            if nearest_group_index is None:
                groups.append([point])
                group_assignment.append(group_index)
                group_index+=1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
        return np.array(group_assignment)
        
    def _determine_nearest_group(self,point,groups):
        nearest_group_index=None
        index=0
        for group in groups:
            dis_fr_group=self._distance_to_group(point,group)
            if dis_fr_group<group_dis_tolerance:
                nearest_group_index=index
            index+=1
        return nearest_group_index
            
    def _distance_to_group(self,point,group):
        min_distance=sys.float_info.max
        for pt in group:
            dist=gmm.euclidean_dist(point,pt)
            if dist<min_distance:
                min_distance=dist
        return min_distance
    
