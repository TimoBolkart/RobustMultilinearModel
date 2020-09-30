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

#include "ExpressionPermutationSA.h"
#include "MathHelper.h"

#include <iostream>

const double MODE_2_REGULARIZER = 0.01;
const double MODE_3_REGULARIZER = 0.01;

//Maximum number of iterations with one temperature / threshold without accepted energy
//const size_t NUM_CONVERGENCE_SUBITER = 20;
const size_t NUM_CONVERGENCE_SUBITER = 50;
//Factor for lowering the threshold for each iteration 
const double THRESHOLD_LOWERING_FACTOR = 0.5; //alpha in (0.0,1.0)
//Factor for the number of iterations with one threshold. Number of iterations increase during optimization
//since lower thresholds require more iteration steps.
const double THRESHOLD_NUM_SUBITER_FACTOR = 0.1;
//Factor for number of samples that are permuted during nearest neighbor sampling.
const double THRESHOLD_NUM_NEIGHBOR_SAMPLE_FACTOR = 0.1;

//#define TMP_DEBUG_OUTPUT

//#define OUTPUT_TIME
#ifdef OUTPUT_TIME
#include "PerformanceCounter.h"
#endif

ExpressionPermutationSA::ExpressionPermutationSA(const std::vector<double>& data, const size_t numIdentities, const size_t numExpressions)
: m_dataSize(data.size())
, m_numIdentities(numIdentities)
, m_numExpressions(numExpressions)
, m_numSamples(numIdentities*numExpressions)
, m_sampleDataDim(m_dataSize/m_numSamples)
, m_numSampleVertices(m_sampleDataDim/3)
{
	//The centering of the data is not affected by shape permutations
	MathHelper::centerData(data, m_sampleDataDim, m_centeredData, m_dataMean);
}

ExpressionPermutationSA::~ExpressionPermutationSA()
{

}

bool ExpressionPermutationSA::optimize(const size_t iIdentity, const double identityWeight, const double expressionWeight, const size_t maxNumIter, const size_t maxNumThresholdIter, const double initThreshold
													, const std::vector<size_t>& initSemCorr, std::vector<size_t>& semCorr)
{

	std::vector<size_t> modeDim;
	modeDim.push_back(m_sampleDataDim);
	modeDim.push_back(m_numIdentities);
	modeDim.push_back(m_numExpressions);

	//Number of neighbors that are permuted randomly
	const size_t numNeighborSamples = std::max<size_t>(static_cast<size_t>(THRESHOLD_NUM_NEIGHBOR_SAMPLE_FACTOR*static_cast<double>(m_numExpressions)+0.5), 1);

	//Current threshold used for acceptance of new values
	double currT = initThreshold;

	//Compute initial energy
	const double initE = ExpressionPermutationSA::computeCompactnessEnergy(m_centeredData, initSemCorr, modeDim, identityWeight, expressionWeight);

	//Minimum energy during optimization
	double minE = initE;
	semCorr = initSemCorr;

	//Current accepted energy during optimization (used for neighbor search of next optimization step)
	double currE = initE;

	//Temporary semantic correspondence that is altered in each step 
	std::vector<size_t> tmpSemCorr = semCorr;

	//Iterate with decreasing the temperature in each iteration
	for(size_t iIter = 0; iIter < maxNumIter; ++iIter)
	{
#ifdef TMP_DEBUG_OUTPUT
		std::cout << "******************************************" << std::endl;
		std::cout << "Iter " << iIter+1 << " of " << maxNumIter << std::endl;
		std::cout << "******************************************" << std::endl;
#endif

		//Iteratively selecte multiple samples for one temperature
		size_t tmpCovergenceIter = 0;

		const size_t currNumThresholdIter = std::min<size_t>(static_cast<size_t>(THRESHOLD_NUM_SUBITER_FACTOR*initThreshold*static_cast<double>(maxNumThresholdIter)/currT+0.5), maxNumThresholdIter);
#ifdef TMP_DEBUG_OUTPUT
		std::cout << "******************************************" << std::endl;
		std::cout << "Num Subiter: " << currNumThresholdIter << std::endl;
		std::cout << "******************************************" << std::endl;
#endif

#ifdef OUTPUT_TIME
		const double thIterStart = PerformanceCounter::getTime();
#endif

		for(size_t iThresIter = 0; iThresIter < currNumThresholdIter; ++iThresIter)
		{
			if(tmpCovergenceIter > NUM_CONVERGENCE_SUBITER)
			{
				break;
			}

#ifdef OUTPUT_TIME
			const double t1 = PerformanceCounter::getTime();
#endif
			//Reset temporary semantic correspondence array
			ExpressionPermutationSA::copyIdentityCorr(semCorr, modeDim, iIdentity, tmpSemCorr);

#ifdef OUTPUT_TIME
			const double t2 = PerformanceCounter::getTime();
#endif

			//Select a random neighbor as new sample
			std::vector<size_t> tmpNeighborPermutation;
			ExpressionPermutationSA::selectRandNeighbor(m_numExpressions, numNeighborSamples, tmpNeighborPermutation);

#ifdef OUTPUT_TIME
			const double t3 = PerformanceCounter::getTime();
#endif

			//Apply the permutation to the temporary correspondence array
			ExpressionPermutationSA::permuteExpressions(tmpSemCorr, modeDim, iIdentity, tmpNeighborPermutation);

#ifdef OUTPUT_TIME
			const double t4 = PerformanceCounter::getTime();
#endif

			//Compute energy of new sample
			const double newE = ExpressionPermutationSA::computeCompactnessEnergy(m_centeredData, tmpSemCorr, modeDim, identityWeight, expressionWeight);

#ifdef OUTPUT_TIME
			const double t5 = PerformanceCounter::getTime();
#endif

#ifdef OUTPUT_TIME
			std::cout << "Reset sem corr " << (t2-t1) << "s" << std::endl;
			std::cout << "Select random neighbor " << (t3-t2) << "s" << std::endl;
			std::cout << "Permute expressions " << (t4-t3) << "s" << std::endl;
			std::cout << "Compute energy " << (t5-t4) << "s" << std::endl;
#endif

			//New energy is smaller -> accept
			//New energy is larger but dE < threshold -> accept
			const double deltaE = newE-currE;

			if(deltaE < currT)
			{
#ifdef TMP_DEBUG_OUTPUT
				std::cout << "Accept" << std::endl;
#endif

				//Accept current permutation and continue with next sample
				currE = newE;
				//Reset convergence iter
				tmpCovergenceIter = 0;

				//If new solution is better than the best solution update the best solution
				if(newE < minE)
				{
					minE = newE;
					ExpressionPermutationSA::copyIdentityCorr(tmpSemCorr, modeDim, iIdentity, semCorr);

#ifdef TMP_DEBUG_OUTPUT
					std::cout << "New minimum: " << minE << std::endl;
#endif
				}

				continue;
			}

#ifdef TMP_DEBUG_OUTPUT
			std::cout << "Reject" << std::endl;
#endif
			++tmpCovergenceIter;
		}

		currT *= THRESHOLD_LOWERING_FACTOR;

#ifdef OUTPUT_TIME
		const double thIterEnd = PerformanceCounter::getTime();
		std::cout << "********************************************************" << std::endl;
		std::cout << "New method: " << std::endl;
		std::cout << "Threshold iter " << (thIterEnd-thIterStart)/60.0 << "min" << std::endl;
		std::cout << "********************************************************" << std::endl;
#endif
		
	}

	return currE < initE;

}

double ExpressionPermutationSA::selectInitialThreshold(const std::vector<size_t>& initSemCorr, const double identityWeight, const double expressionWeight)
{
	const double idPercentage = 10.0;
	const double factor = 0.5;

	std::vector<size_t> modeDim;
	modeDim.push_back(m_sampleDataDim);
	modeDim.push_back(m_numIdentities);
	modeDim.push_back(m_numExpressions);

	//Compute initial energy
	const double initE = ExpressionPermutationSA::computeCompactnessEnergy(m_centeredData, initSemCorr, modeDim, identityWeight, expressionWeight);

	std::vector<int> randIdentities;
	MathHelper::getRandomlyPermutedInteger(0, static_cast<int>(m_numIdentities-1), randIdentities);

	double initThreshold(0.0);

	//Temporary semantic correspondence that is altered in each step 
	std::vector<size_t> tmpSemCorr = initSemCorr;

	const size_t numRandIdentities = std::min<size_t>(std::max<size_t>(static_cast<size_t>((idPercentage/100.0)*static_cast<double>(m_numIdentities)+0.5),1),m_numIdentities);
	for(size_t i = 0; i < numRandIdentities; ++i)
	{
		const size_t currId = randIdentities[i];

		//Compute random permutation for current identity
		std::vector<int> randExpInts;
		MathHelper::getRandomlyPermutedInteger(0, static_cast<int>(m_numExpressions-1), randExpInts);

		std::vector<size_t> randPermutation;
		randPermutation.resize(m_numExpressions);
		for(size_t j = 0; j < m_numExpressions; ++j)
		{
			randPermutation[j] = static_cast<size_t>(randExpInts[j]);
		}

		//Apply the permutation to the temporary correspondence array
		ExpressionPermutationSA::permuteExpressions(tmpSemCorr, modeDim, currId, randPermutation);

		//Compute energy
		const double randE = ExpressionPermutationSA::computeCompactnessEnergy(m_centeredData, tmpSemCorr, modeDim, identityWeight, expressionWeight);
		initThreshold += fabs(randE-initE);

		//Reset current identity
		ExpressionPermutationSA::copyIdentityCorr(initSemCorr, modeDim, currId, tmpSemCorr);
	}

	return factor*initThreshold;
}

double ExpressionPermutationSA::computeCompactnessEnergy(const std::vector<double>& centeredData, const std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const double identityWeight, const double expressionWeight)
{
	const double e2 = ExpressionPermutationSA::computeModeCompactnessEnergy(centeredData, semCorr, modeDim, 2, identityWeight);
	const double e3 = ExpressionPermutationSA::computeModeCompactnessEnergy(centeredData, semCorr, modeDim, 3, expressionWeight);
	return e2+e3;
}

double ExpressionPermutationSA::computeModeCompactnessEnergy(const std::vector<double>& centeredData, const std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const size_t mode, const double modeWeight)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
		return 0.0;
	}

	std::vector<double> modeEigenValues;
	ExpressionPermutationSA::computeCovarianceEigenvalues(centeredData, semCorr, modeDim, mode, modeEigenValues);

	const size_t numModeEigenValues = modeEigenValues.size();

	bool bMode2 = (mode == 2);
	const double modeRegularizer = bMode2 ? MODE_2_REGULARIZER : MODE_3_REGULARIZER;

	//Compute energy
	double value(0.0);
	for(size_t i = 0; i < numModeEigenValues; ++i)
	{
		value += log(modeEigenValues[i] + modeRegularizer);
		//value -= log(modeRegularizer);
	}

	const double currModeWeight = modeWeight/static_cast<double>(numModeEigenValues);
	return currModeWeight*value;
}

void ExpressionPermutationSA::computeCovarianceEigenvalues(const std::vector<double>& centeredData, const std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const size_t mode, std::vector<double>& eigenValues)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "ExpressionPermutationCostFunction::computeCovarianceEigenVectors(...) - wrong mode input " << mode << std::endl;
		return;
	}

	size_t dim(0);
	std::vector<double> covarianceMatrix;
	if(!MathHelper::computeCovarianceMatrix(centeredData, semCorr, modeDim, mode, covarianceMatrix, dim))
	{
		std::cout << "Failed computing covariance matrix" << std::endl;
		return;
	}

	std::vector<double> tmpEigenValues;
	MathHelper::computeSingularValues(covarianceMatrix, dim, dim, tmpEigenValues);
	
	size_t numEigenValues(0);
	for(size_t i = 0; i < tmpEigenValues.size(); ++i)
	{
		if(fabs(tmpEigenValues[i]) > math_eps)
		{
			++numEigenValues;
		}
	}

	eigenValues.resize(numEigenValues);

#pragma omp parallel for
	for(int i = 0; i < numEigenValues; ++i)
	{
		eigenValues[i] = tmpEigenValues[i];
	}
}

void ExpressionPermutationSA::permuteExpressions(std::vector<size_t>& semCorr, const std::vector<size_t>& modeDim, const size_t iIdentity, const std::vector<size_t>& expressionPermutation)
{
	const size_t d2 = modeDim[1];
	const size_t d3 = modeDim[2];
	if(expressionPermutation.size() != d3)
	{
		std::cout << "Wrong expression permutation array dimension" << std::endl;
		return;
	}

	std::vector<size_t> tmpCorr;
	tmpCorr.resize(d3);

	for(size_t i = 0; i < d3; ++i)
	{
		const size_t index1 = i*d2 + iIdentity;
		tmpCorr[i] = semCorr[index1];
	}

	for(size_t i = 0; i < d3; ++i)
	{
		const size_t index2 = expressionPermutation[i]*d2 + iIdentity;
		semCorr[index2] = tmpCorr[i];
	}
}

void ExpressionPermutationSA::copyIdentityCorr(const std::vector<size_t>& sourceSemCorr, const std::vector<size_t>& modeDim, const size_t iIdentity, std::vector<size_t>& targetSemCorr)
{
	const size_t d2 = modeDim[1];
	const size_t d3 = modeDim[2];
	
	for(size_t i = 0; i < d3; ++i)
	{
		const size_t index = i*d2 + iIdentity;
		targetSemCorr[index] = sourceSemCorr[index];
	}
}

void ExpressionPermutationSA::selectRandNeighbor(const size_t dim, const size_t numSwaps, std::vector<size_t>& randNeighborPermutation)
{
	randNeighborPermutation.clear();
	randNeighborPermutation.resize(dim);

	for(size_t i = 0; i < dim; ++i)
	{
		randNeighborPermutation[i] = i;
	}

	for(size_t i = 0; i < numSwaps; ++i)
	{
		std::vector<int> randInts;
		MathHelper::getRandomlyPermutedInteger(0, static_cast<int>(dim-1), randInts);

		const size_t value1 = randNeighborPermutation[randInts[0]];
		const size_t value2 = randNeighborPermutation[randInts[1]];
		randNeighborPermutation[randInts[1]] = value1;
		randNeighborPermutation[randInts[0]] = value2;
	}
}
