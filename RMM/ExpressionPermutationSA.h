/*************************************************************************************************************************/
// This source is provided for NON-COMMERCIAL RESEARCH PURPOSES only, and is provided “as is” WITHOUT ANY WARRANTY; 
// without even the implied warranty of fitness for a particular purpose. The redistribution of the code is not permitted.
//
// If you use the source or part of it in a publication, cite the following paper:
// 
// T. Bolkart, S. Wuhrer
// A Robust Multilinear Model Learning Framework for 3D Faces.
// Conference on Computer Vision and Pattern Recognition (CVPR), 2016
//
// Copyright (c) 2016 Timo Bolkart, Stefanie Wuhrer
/*************************************************************************************************************************/

#ifndef EXPRESSIONPERMUTATIONSA_H
#define EXPRESSIONPERMUTATIONSA_H

#include <vector>
#include <stdlib.h>

class ExpressionPermutationSA
{
public:
	ExpressionPermutationSA(const std::vector<double>& data, const size_t numIdentities, const size_t numExpressions);
	
	~ExpressionPermutationSA();	
	
	bool optimize(const size_t iIdentity, const double identityWeight, const double expressionWeight, const size_t maxNumIter, const size_t maxNumThresholdIter, const double initThreshold
					, const std::vector<size_t>& initSemCorr, std::vector<size_t>& semCorr);

	double selectInitialThreshold(const std::vector<size_t>& initSemCorr, const double identityWeight, const double expressionWeight);	

private:

	static double computeCompactnessEnergy(const std::vector<double>& centeredData, const std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const double identityWeight, const double expressionWeight);

	static double computeModeCompactnessEnergy(const std::vector<double>& centeredData, const std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const size_t mode, const double modeWeight);

	static void computeCovarianceEigenvalues(const std::vector<double>& centeredData, const std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const size_t mode, std::vector<double>& eigenValues);

	static void computeCovarianceMatrix(const std::vector<double>& centeredData, const std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const size_t mode, std::vector<double>& covarianceMatrix, size_t& matrixDim);
	
	static void permuteExpressions(std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const size_t iIdentity, const std::vector<size_t>& expressionPermutation);
	
	static void copyIdentityCorr(const std::vector<size_t>& sourceSemCorr, const std::vector<size_t>& modeDim, const size_t iIdentity, std::vector<size_t>& targetSemCorr);

	static void selectRandNeighbor(const size_t dim, const size_t numSwaps, std::vector<size_t>& randNeighborPermutation);


	ExpressionPermutationSA(const ExpressionPermutationSA&);

	ExpressionPermutationSA& operator=(const ExpressionPermutationSA&);


	//Dimension d1*d2*d3
	std::vector<double> m_centeredData;
	//Dimension d1
	std::vector<double> m_dataMean;	

	const size_t m_dataSize;
	const size_t m_numIdentities;
	const size_t m_numExpressions;
	const size_t m_numSamples;
	const size_t m_sampleDataDim;
	const size_t m_numSampleVertices;
};

#endif
