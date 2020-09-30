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

#ifndef KDTREE3_H
#define KDTREE3_H

#include <ANN\ANN.h>

#include <vector>

class KDTree3
{
public:
	KDTree3(const std::vector<double>& points);

	~KDTree3();

	bool getNearestPoint(const std::vector<double>& point, int& pointIndex, double& sqrDist) const;

	bool getKNearestPoints(const std::vector<double>& point, const size_t k, std::vector<int>& pointIndexVec, std::vector<double>& sqrDistVec) const;

private:
	KDTree3(const KDTree3& kdTree);
	
	KDTree3& operator=(const KDTree3& kdTree);

	ANNpointArray m_pointArray;

	ANNkd_tree* m_pKDTree;
};

#endif