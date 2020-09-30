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

#include "RelabelMultilinearData.h"
#include "MathHelper.h"
#include "ExpressionPermutationSA.h"
#include "PerformanceCounter.h"
#include "FileLoader.h"
#include "FileWriter.h"
#include "MDLHelper.h"

#include <fstream>
#include <iostream>

#define OUTPUT_TIME
#define OUTPUT_COMPACTNESS
#define OUTPUT_CONFIG
#define OUTPUT_PERMUTATION_ITER

#ifdef OUTPUT_TIME
#include "PerformanceCounter.h"
#endif

//const size_t NUM_IDENTITY_ITER = 15;
const size_t NUM_IDENTITY_ITER = 15;
const size_t NUM_SA_ITER = 10;
const size_t NUM_SA_SUBITER = 200;

bool RelabelMultilinearData::optimizeSemanticCorrespondence(const std::vector<double>& data, const std::vector<size_t>& modeDim, const size_t numIter, const std::vector<size_t>& initCorr, std::vector<size_t>& semCorr)
{
	const size_t d2 = modeDim[1];
	const size_t d3 = modeDim[2];
	const size_t numShapes = d2*d3;

	//Initialize semantic correspondence
	semCorr.clear();
	semCorr = initCorr;

	ExpressionPermutationSA expPermutSA(data, d2, d3);

	//Select initial threshold
	const double initT = expPermutSA.selectInitialThreshold(initCorr, 1.0, 1.0);

	for(size_t iIter = 0; iIter < numIter; ++iIter)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "++++++++++++++++++++++++++++++++" << std::endl;
		std::cout << "Sem. Corr. Iteration " << iIter+1 << std::endl;
		std::cout << "++++++++++++++++++++++++++++++++" << std::endl;
#endif

		size_t numPermutedIds(0);

		//Optimize semantic correspondence for each identity by re-parametrizing its expressions
		std::vector<int> randomIntegers;
		MathHelper::getRandomlyPermutedInteger(0, static_cast<int>(d2-1), randomIntegers);

		for(size_t iId = 0; iId < d2; ++iId)
		{
			const size_t currIdentity = randomIntegers[iId];	

#ifdef DEBUG_OUTPUT
			std::cout << "Processing identity " << currIdentity << "(" << iId+1 << " of " << d2 << ")" << std::endl;
#endif

			//Optimize semantic correspondence for current identity
			std::vector<size_t> optSemCorr;
			if(!expPermutSA.optimize(currIdentity, 1.0, 1.0, NUM_SA_ITER, NUM_SA_SUBITER, initT, semCorr, optSemCorr))
			{
				std::cout << "No optimization for identity " << currIdentity << std::endl;
				continue;
			}

			++numPermutedIds;

			//Set the optimized values in the case of success
			for(size_t iExp = 0; iExp < d3; ++iExp)
			{
				const size_t index = iExp*d2 + currIdentity;
				semCorr[index] = optSemCorr[index];
			}
		}

#ifdef DEBUG_OUTPUT
		std::cout << "Num permuted identities: " << numPermutedIds << std::endl;
#endif

		if(numPermutedIds == 0)
		{
			break;
		}
	}

	return true;
}

void RelabelMultilinearData::outputPermutationMatrix(const std::vector<size_t>& permutations, const size_t numRows, const size_t numCols, const std::string& sstrOutFileName)
{
	if(permutations.size() != numRows*numCols)
	{
		return;
	}

	std::fstream outStream(sstrOutFileName, std::ios::out);
	for(size_t iRow = 0; iRow < numRows; ++iRow)
	{
		for(size_t iCol = 0; iCol < numCols; ++iCol)
		{
			outStream << permutations[iCol*numRows+iRow] << " ";
		}
		outStream << std::endl;
	}
	outStream.close();
}