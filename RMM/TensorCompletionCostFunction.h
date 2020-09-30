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

#ifndef TENSORCOMPLETIONCOSTFUNCTION_H
#define TENSORCOMPLETIONCOSTFUNCTION_H

#include "Definitions.h"
#include "DataContainer.h"

#include <vnl/vnl_vector.h>
#include <vnl/vnl_cost_function.h>

#include <vector>
#include <set>
#include <map>

class KDTree3;

class TensorCompletionCostFunction : public vnl_cost_function
{
public:
	TensorCompletionCostFunction(const std::vector<double>& data, const size_t numIdentities, const size_t numExpressions, const std::vector<size_t>& semCorr
											, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const double identityWeight, const double expressionWeight, const size_t iShape);

	TensorCompletionCostFunction(const std::vector<double>& data, const size_t numIdentities, const size_t numExpressions, const std::vector<size_t>& semCorr
											, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes
											, const std::vector<double>& corrShapeData, const std::vector<double>& corrShapeNormals, const KDTree3& corrShapeKDTree, const double maxCorrDataDist
											, const double& corrShape_s, const std::vector<double>& corrShape_R, const std::vector<double>& corrShape_t
											, const double identityWeight, const double expressionWeight, const double corrDataWeight, const size_t iShape);

	~TensorCompletionCostFunction();

	bool setSmoothnessValues(const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const double smoothnessWeight);

	bool setSmoothnessValues(const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const double smoothnessWeight, const std::vector<double>& vertexLaplacian);

	virtual void compute(const vnl_vector<double>& x, double* f, vnl_vector<double>* g);

private:
	void init(double* f, vnl_vector<double>* g);

	void updateShapeData(const vnl_vector<double>& x);

	void addModeCompactnessEnergy(const size_t mode, const std::vector<double>& modeEigenValues, double* f);
	
	void addModeShapeCompactnessGradient(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, vnl_vector<double>* g);

	void addCorruptShapeDataEnergy(double* f, vnl_vector<double>* g);

	void addShapeSmoothnessEnergy(double* f, vnl_vector<double>* g);

	void computeEigenValueDerivative(const size_t mode, const std::vector<double>& precomputedModeDataSum, const std::vector<double>& modeEigenVectors, const size_t eigenValueIndex
												, const size_t iIndex, const size_t jIndex, std::vector<double>& dLambda_dXij);

	void computeModeDataSum(const size_t mode, std::vector<double>& modeDataSum);

	void computeCovarianceEigenVectors(const size_t mode, std::vector<double>& eigenVectors, std::vector<double>& eigenValues);

	bool checkCurruptShapeData();

	TensorCompletionCostFunction(const TensorCompletionCostFunction& costFunction);

	TensorCompletionCostFunction& operator=(const TensorCompletionCostFunction& costFunction);

	//Vertices of all samples (d2 identities, d3 expressions)
	//Changed w.r.t. the current computed parametrization within each iteration step
	//Dimension d1*d2*d3
	const std::vector<double>& m_data;
	std::vector<double> m_updateData;
	//Centered data
	//Dimension d1*d2*d3
	std::vector<double> m_centeredData;
	//Dimension d1
	std::vector<double> m_dataMean;

	const std::vector<size_t>& m_semCorr;

	//Data necessary if partial data are given 
	const std::vector<double>* m_pCorrShapeData;
	const std::vector<double>* m_pCorrShapeNormals;
	const KDTree3* m_pCorrShapeKDTree; 
	const double m_maxCorrDataDist;
	const double m_corrShape_s;
	const std::vector<double> m_corrShape_R;
	const std::vector<double> m_corrShape_t;

	//Data necessary for smoothing regularization (e.g. bi-Laplace)
	std::vector<std::vector<size_t>> m_precomputedSmoothnessIndices;
	std::vector<std::vector<double>> m_precomputedSmoothnessWeights;
	std::vector<double> m_vertexLaplacian;

	const size_t m_dataSize;
	const size_t m_numIdentities;
	const size_t m_numExpressions;
	const size_t m_numSamples;
	const size_t m_sampleDataDim;
	const size_t m_numSampleVertices;
	
	const	double m_identityWeight;
	const double m_expressionWeight;
	const double m_corrDataWeight;
	double m_smoothWeight;

	const size_t m_iIndex;
	const size_t m_jIndex;
	const size_t m_shapeIndex;
	const size_t m_shapeOffset;

	bool m_bMissingShape;
	bool m_bCurruptShape;
	bool m_bUpdateData;
};

#endif