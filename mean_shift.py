import numpy as np
from point_ref import point_Grouper
import gmm as gm
MIN_DISTANCE=0.000001
class MeanShift(object):
    def __init__(self, kernel=gm.gaussian_kernal):
        if kernal=='multivariate_gaussian':
            kernal=gm.multivariate_gaussian_kernal
        self.kernal=kernal
def cluster(self,points,kernel_bandwidth,iteration_callback=None):
    if (iteration_callback):
        iteration_callback(points,0)

        shift_points=np.array(points)
        max_min_dist=1
        iteration_number=0

        still_shifting=[True]*points.shape[0]

        while max_min_dist>MIN_DISTANCE:
            
            max_min_dist=0
            iteration_number+=1
            for i in range(0,len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new=shift_points[i]
                p_new_start=p_new
                p_new=self._shift_point(p_new, points, kernel_bandwidth)
                dist=gm.euclidean_dist(p_new,p_new_start)
                if dist >max_min_dist:
                    max_min_dist=dist
                if dist<MIN_DISTANCE:
                    still_shifting[i]=False
                shift_points[i]=p_new
            if (iteration_callback):
                iteration_callback(shift_points,iteration_number)
            point_grouper=point_Grouper()
            group_assignments=point_grouper.group_points(shift_points.tolist())
            return MeanShiftResult(points, shift_points, group_assignments)
def _shift_point(self, point, points, kernel_bandwidth):
    points = np.array(points)
    point_weights = self.kernel(point-points, kernel_bandwidth)
    tiled_weights = np.tile(point_weights, [len(point), 1])
    denominator = sum(point_weights)
    shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
    return shifted_point
                    
                    
        
