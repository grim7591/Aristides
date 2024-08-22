from RegressionOne import run_regression, evaluate_model
from PreProcessingFunctions import preprocess_data
from IAAOFunctions import PRD, PRB, COD, weightedMean, averageDeviation
import numpy as np

data_path = 'Data/dp22.csv'
market_areas_path = 'Data/normalizedMAs.csv'

data = preprocess_data(data_path, market_areas_path)

formula = """
Assessment_Val ~ living_area + legal_acreage + percent_good +
ALACHUA + ARCHER + GAINESVILLE + HAWTHORNE + HIGH_SPRINGS + NEWBERRY + 
WALDO + Springtree_B + HighSprings_A + MidtownEast_C + swNewberry_B + 
MidtownEast_A + swNewberry_A + MidtownEast_B + HighSprings_F + WaldoRural_C +
Springtree_A + Tioga_B + Tioga_A + swNewberry_C + MidtownEast_D + HighSprings_E +
MidtownEast_E + HighSprings_D + Springtree_C + WaldoRural_A + WaldoRural_B + 
HighSprings_C + MidtownEast_F + in_subdivision + imprv_det_quality_cd + total_porch_area + total_garage_area
"""

regresult, test_data = run_regression(data,formula)
evaluate_model(regresult, test_data)