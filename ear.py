from scipy.spatial import distance

def calculateEAR(P):
	verticalDist1 = distance.euclidean(P[1], P[5])
	verticalDist2 = distance.euclidean(P[2], P[4])
	horizontalDist = distance.euclidean(P[0], P[3])

	ear = (verticalDist1 + verticalDist2)/(2.0*horizontalDist) 
	return ear
