import measures.zliobaite_measures as measures

def printAbsoluteMeasures(outcomes, protected):
	fairness = measures.elift(outcomes, protected)
	print('elift ratio: ', fairness)

	fairness = measures.odds_ratio(outcomes, protected)
	print('odds ratio: ', fairness)

	fairness = measures.impact_ratio(outcomes, protected)
	print('impact ratio: ', fairness)

	fairness = measures.mean_difference(outcomes, protected)
	print('mean difference: ', fairness)

	fairness = measures.normalized_difference(outcomes, protected)
	print('normalized difference: ', fairness)