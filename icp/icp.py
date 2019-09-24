import numpy as np
import pdb
import pandas as pd

class ICP(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        if self.src.shape[0] > self.dst.shape[0]:
            self.src = self.src[ : self.dst.shape[0], :]
        else:
            self.dst = self.dst[ : self.src.shape[0], :]
        
        assert self.src.shape[0] == self.dst.shape[0]
        self.result = np.zeros_like(self.src)

    def normalize_pts(self, pts):
        pts_mean = np.mean(pts, axis=0)
        norm_pts = pts - pts_mean
        return norm_pts

    def getPointCloudRegistration(self, target):
        # assert self.result and target
        assert self.result.shape[0] == target.shape[0]
        norm_result = self.normalize_pts(self.result)
        norm_target = self.normalize_pts(target)
        H = np.dot(norm_result.T, norm_target)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        # Consider reflection case
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(Vt.T, U.T)
        # Raise a warning if the SVD is not correct
        if abs(np.linalg.det(R) - 1.0) > 0.0001:
            Warning("Direct Point Cloud registration unstable!")
        T = np.mean(target, axis=0) - np.mean(self.result, axis=0).dot(R.T)
        return R, T        


    def computeCorrespondence(self):
        # assert self.result and self.dst
        assert self.result.shape[0] == self.dst.shape[0]
        indexArray = np.zeros(self.result.shape[0], dtype=np.int)
        totalDistance = 0.0
        for resultIndex, resultValue in enumerate(self.result):
            minIndex, minDistance = -1, np.inf
            for dstIndex, dstValue in enumerate(self.dst):
                distance = np.linalg.norm(resultValue - dstValue)
                if distance < minDistance:
                    minDistance, minIndex = distance, dstIndex
            indexArray[resultIndex] = minIndex
            totalDistance += minDistance

        return totalDistance, self.dst[indexArray, 0:3]
    
    def solve(self):
        print ('Solve ICP with', self.__class__.__name__)
        distanceThres, maxIteration, iteration = 0.001, 20, 0
        # Perform the initial computation of correspondence
        currentDistance, target = self.computeCorrespondence()
        print ("Init ICP, distance: %f" % currentDistance)
        # ICP loop
        while currentDistance > distanceThres and iteration < maxIteration:
            # Compute tranformation between self.result and self.target
            R, T = self.getPointCloudRegistration(target)

            # Appy transformation to self.result
            self.result = self.result.dot(R.T) + T.T
            self.result = self.normalize_pts(self.result)

            # Compute point correspondence
            currentDistance, target = self.computeCorrespondence()

            # Update
            iteration += 1
            print ("Iteration: %5d, with total distance: %f" % (iteration, currentDistance))

if __name__ == '__main__':

    pc1 = pd.read_csv('/deepSDF/train_data/03790512/000159.pts', sep=" ", header = None).values
    pc2 = pd.read_csv('/deepSDF/train_data/03790512/000144.pts', sep=" ", header = None).values
    icp = ICP(pc1, pc2)
    icp.solve()