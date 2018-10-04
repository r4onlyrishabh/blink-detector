from scipy.spatial.distance import euclidean

def calculateEAR(P):
	verticalDist1 = euclidean(P[1], P[5])
	verticalDist2 = euclidean(P[2], P[4])
	horizontalDist = euclidean(P[0], P[3])

	ear = (verticalDist1 + verticalDist2)/(2.0*horizontalDist) 
	return ear
