from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker():
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
		    # when registering an object we use the next available object
		    # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                  self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame  
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

		    # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid   
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

		    # if we are currently not tracking any objects take the input
		    # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

		    # otherwise, are are currently tracking objects so we need to
		    # try to match the input centroids to existing object
		    # centroids
        else:
			      # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # closest distance pair of centroids to each other and
            # (2) ensure that the distance value is less than the
            # maximum distance
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
			      # list
            rows = D.min(axis=1).argsort()
            # in the first loop, we'll find the largest value in each row
            # and then sort the row indexes based on their max values,
            # allowing us to build our first tentative assignment of
            # object IDs to input centroids
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                currentCentroid = self.objects[objectID]
                inputCentroid = inputCentroids[col]
                
                if abs(currentCentroid[0] - inputCentroid[0]) <= 200:
                    self.objects[objectID] = inputCentroid
                    self.disappeared[objectID] = 0
                    # indicate that we have examined each of the row and
                    # column indexes, respectively
                #inputCentroid[0]>1000　　delete inputcentroid[0]
                if inputCentroid[0]>1000:
                    self.deregister(objectID)                
                    usedRows.add(row)
                    usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            #Conditions for the same objectID when the x-coordinates of inputCentroids and objectCentroids are within the range of 200
            if D.shape[0] >= D.shape[1]:
        
			          # in the event that the number of object centroids is
                # equal or greater than the number of input centroids
                # we need to check and see if some of these objects have
			          # potentially disappeared

				        # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
		    # return the set of trackable objects
        return self.objects