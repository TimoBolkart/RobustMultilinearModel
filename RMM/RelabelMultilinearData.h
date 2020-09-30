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

#ifndef RELABELMULTILINEARDATA_H
#define RELABELMULTILINEARDATA_H

#include <vector>
#include <set>
#include <stdlib.h>
#include <string>

class RelabelMultilinearData
{
public:
	static bool optimizeSemanticCorrespondence(const std::vector<double>& data, const std::vector<size_t>& modeDim, const size_t numIter, const std::vector<size_t>& initCorr, std::vector<size_t>& semCorr);

	static void outputPermutationMatrix(const std::vector<size_t>& permutations, const size_t numRows, const size_t numCols, const std::string& sstrOutFileName);
};

#endif
