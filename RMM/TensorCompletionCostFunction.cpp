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

#include "TensorCompletionCostFunction.h"
#include "MathHelper.h"
#include "KDTree3.h"
#include "VectorNX.h"

#include <iostream>

#define USE_POINT_PLANE_CORRESPONDENCE

//If enabled, vertices with distance to nearest neighbor larger than threshold are ignored
//If disabled, vector to nearest neighbor is scaled to at most threshold length
//#define NN_DIST_THRESHOLD_METHOD

//#define OUTPUT_FUNCTION_VALUE

#ifdef OUTPUT_FUNCTION_VALUE
#include <iomanip>
#endif

const double MODE_2_REGULARIZER = 0.01;
const double MODE_3_REGULARIZER = 0.01;
const double CORR_ENGERY_REGULARIZER = 1.0;

TensorCompletionCostFunction::TensorCompletionCostFunction(const std::vector<double>& data, const size_t numIdentities, const size_t numExpressions, const std::vector<size_t>& semCorr
																			, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const double identityWeight, const double expressionWeight, const size_t iShape)
: vnl_cost_function(data.size()/(numIdentities*numExpressions))
, m_data(data)
, m_updateData(data)
, m_semCorr(semCorr)
, m_pCorrShapeData(NULL)
, m_pCorrShapeNormals(NULL)
, m_pCorrShapeKDTree(NULL) 
, m_maxCorrDataDist(0.0)
, m_corrShape_s(0.0)
, m_corrShape_R(std::vector<double>())
, m_corrShape_t(std::vector<double>())
, m_precomputedSmoothnessIndices()
, m_precomputedSmoothnessWeights()
, m_dataSize(data.size())
, m_numIdentities(numIdentities)
, m_numExpressions(numExpressions)
, m_numSamples(numIdentities*numExpressions)
, m_sampleDataDim(m_dataSize/m_numSamples)
, m_numSampleVertices(m_sampleDataDim/3)
, m_identityWeight(identityWeight)
, m_expressionWeight(expressionWeight)
, m_corrDataWeight(0.0)
, m_smoothWeight(0.0)
, m_iIndex(iShape%numIdentities)
, m_jIndex((iShape-m_iIndex)/numIdentities)
, m_shapeIndex(semCorr[iShape])
, m_shapeOffset(m_shapeIndex*m_sampleDataDim)
, m_bMissingShape(missingShapes[m_shapeIndex])
, m_bCurruptShape(corruptShapes[m_shapeIndex])
, m_bUpdateData(false)
{

}

TensorCompletionCostFunction::TensorCompletionCostFunction(const std::vector<double>& data, const size_t numIdentities, const size_t numExpressions, const std::vector<size_t>& semCorr
																			, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes
																			, const std::vector<double>& corrShapeData, const std::vector<double>& corrShapeNormals, const KDTree3& corrShapeKDTree, const double maxCorrDataDist
																			, const double& corrShape_s, const std::vector<double>& corrShape_R, const std::vector<double>& corrShape_t
																			, const double identityWeight, const double expressionWeight, const double corrDataWeight, const size_t iShape)
: vnl_cost_function(data.size()/(numIdentities*numExpressions))
, m_data(data)
, m_updateData(data)
, m_semCorr(semCorr)
, m_pCorrShapeData(&corrShapeData)
, m_pCorrShapeNormals(&corrShapeNormals)
, m_pCorrShapeKDTree(&corrShapeKDTree) 
, m_maxCorrDataDist(maxCorrDataDist)
, m_corrShape_s(corrShape_s)
, m_corrShape_R(corrShape_R)
, m_corrShape_t(corrShape_t)
, m_precomputedSmoothnessIndices()
, m_precomputedSmoothnessWeights()
, m_dataSize(data.size())
, m_numIdentities(numIdentities)
, m_numExpressions(numExpressions)
, m_numSamples(numIdentities*numExpressions)
, m_sampleDataDim(m_dataSize/m_numSamples)
, m_numSampleVertices(m_sampleDataDim/3)
, m_identityWeight(identityWeight)
, m_expressionWeight(expressionWeight)
, m_corrDataWeight(corrDataWeight)
, m_smoothWeight(0.0)
, m_iIndex(iShape%numIdentities)
, m_jIndex((iShape-m_iIndex)/numIdentities)
, m_shapeIndex(semCorr[iShape])
, m_shapeOffset(m_shapeIndex*m_sampleDataDim)
, m_bMissingShape(missingShapes[m_shapeIndex])
, m_bCurruptShape(corruptShapes[m_shapeIndex])
, m_bUpdateData(false)
{
	bool bCheck = checkCurruptShapeData();
	if(!bCheck)
	{
		std::cout << "Invalid corrupt shape data" << std::endl;
		m_bCurruptShape = false;
	}
}

TensorCompletionCostFunction::~TensorCompletionCostFunction()
{

}

bool TensorCompletionCostFunction::setSmoothnessValues(const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const double smoothnessWeight)
{
	if(precomputedSmoothnessIndices.size() != precomputedSmoothnessWeights.size())
	{
		return false;
	}

	m_precomputedSmoothnessIndices = precomputedSmoothnessIndices;
	m_precomputedSmoothnessWeights = precomputedSmoothnessWeights;
	m_smoothWeight = smoothnessWeight;
	m_vertexLaplacian.clear();
	return true;
}

bool TensorCompletionCostFunction::setSmoothnessValues(const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const double smoothnessWeight
																		, const std::vector<double>& vertexLaplacian)
{
	if(precomputedSmoothnessIndices.size() != precomputedSmoothnessWeights.size())
	{
		return false;
	}

	m_precomputedSmoothnessIndices = precomputedSmoothnessIndices;
	m_precomputedSmoothnessWeights = precomputedSmoothnessWeights;
	m_smoothWeight = smoothnessWeight;
	m_vertexLaplacian = vertexLaplacian;
	return true;
}

void TensorCompletionCostFunction::compute(const vnl_vector<double>& x, double* f, vnl_vector<double>* g)
{
	//Initialize gradient and function
	init(f, g);

	//If current shape is not one of the missing shapes return
	if(!m_bMissingShape && !m_bCurruptShape)
	{
		return;
	}

	updateShapeData(x);
	MathHelper::centerData(m_updateData, m_sampleDataDim, m_centeredData, m_dataMean);

	std::vector<double> mode2EigenVectors; 
	std::vector<double> mode2EigenValues;
	computeCovarianceEigenVectors(2, mode2EigenVectors, mode2EigenValues);

	std::vector<double> mode3EigenVectors; 
	std::vector<double> mode3EigenValues;
	computeCovarianceEigenVectors(3, mode3EigenVectors, mode3EigenValues);

#ifdef OUTPUT_FUNCTION_VALUE
	double f_prev = *f;
#endif

	addModeCompactnessEnergy(2, mode2EigenValues, f);

#ifdef OUTPUT_FUNCTION_VALUE
	double f_id = *f-f_prev;
#endif

	addModeCompactnessEnergy(3, mode3EigenValues, f);

#ifdef OUTPUT_FUNCTION_VALUE
	double f_exp = *f-f_id;
#endif

	addModeShapeCompactnessGradient(2, mode2EigenVectors, mode2EigenValues, g);
	addModeShapeCompactnessGradient(3, mode3EigenVectors, mode3EigenValues, g);

#ifdef OUTPUT_FUNCTION_VALUE
	f_prev = *f;
	double f_corr_data = 0.0;
	double f_corr_lmk = 0.0;
#endif

	if(m_bCurruptShape)
	{
		addCorruptShapeDataEnergy(f, g);

#ifdef OUTPUT_FUNCTION_VALUE
		f_corr_data = *f-f_prev;
		f_prev = *f;
#endif
	}

#ifdef OUTPUT_FUNCTION_VALUE
	f_prev = *f;
#endif

	addShapeSmoothnessEnergy(f, g);

#ifdef OUTPUT_FUNCTION_VALUE
	double f_smooth = *f-f_prev;
	f_prev = *f;
#endif

#ifdef OUTPUT_FUNCTION_VALUE
	std::cout << std::endl;
	std::cout << "E_comp: " <<  std::setprecision(5) << f_id+f_exp << " (Id: " << f_id << " Exp: " << f_exp << ")" << std::endl;
	std::cout << "E_corr: " <<  std::setprecision(5) << f_corr_data+f_corr_lmk << " (Data: " << f_corr_data << " Lmk: " << f_corr_lmk << ")" << std::endl;
	std::cout << "E_smooth: " <<  std::setprecision(5) << f_smooth << std::endl;
	std::cout << "Function value: " << std::setprecision(5) << *f << std::endl;
	std::cout << std::endl;
#endif
}

void TensorCompletionCostFunction::init(double* f, vnl_vector<double>* g)
{
	*f = 0.0;
		
#pragma omp parallel for
	for(int i = 0; i < m_sampleDataDim; ++i)
	{
		(*g)[i] = 0.0;
	}
}

void TensorCompletionCostFunction::updateShapeData(const vnl_vector<double>& x)
{
	for(int i = 0; i < m_sampleDataDim; ++i)
	{
		m_updateData[m_shapeOffset+i] = m_data[m_shapeOffset+i]+x[i];
	}
}

void TensorCompletionCostFunction::addModeCompactnessEnergy(const size_t mode, const std::vector<double>& modeEigenValues, double* f)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
		return;
	}

	const size_t numModeEigenValues = modeEigenValues.size();

	bool bMode2 = (mode == 2);
	const double modeWeight = (bMode2 ? m_identityWeight : m_expressionWeight)/static_cast<double>(numModeEigenValues);
	const double modeRegularizer = bMode2 ? MODE_2_REGULARIZER : MODE_3_REGULARIZER;

	if(modeWeight < math_eps)
	{
		return;
	}

	//Compute energy
	double value(0.0);
	for(size_t i = 0; i < numModeEigenValues; ++i)
	{
		value += log(modeEigenValues[i] + modeRegularizer);
		//value -= log(modeRegularizer);
	}

	value *= modeWeight;
	(*f) += value;
}

void TensorCompletionCostFunction::addModeShapeCompactnessGradient(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, vnl_vector<double>* g)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
		return;
	}

	const size_t numModeEigenValues = modeEigenValues.size();

	bool bMode2 = (mode == 2);
	const double modeWeight = (bMode2 ? m_identityWeight : m_expressionWeight)/static_cast<double>(numModeEigenValues);
	const double modeRegularizer = bMode2 ? MODE_2_REGULARIZER : MODE_3_REGULARIZER;

	if(modeWeight < math_eps)
	{
		return;
	}

	//Optimized version
	std::vector<double> tmpMemoryMatrix;
	tmpMemoryMatrix.resize(m_sampleDataDim * numModeEigenValues, 0.0);

	//Pre-compute the sum over all expressions (mode == 2) or identities (mode == 3)
	std::vector<double> modeDataSum;
	computeModeDataSum(mode, modeDataSum);

#pragma omp parallel for
	for(int a = 0; a < numModeEigenValues; ++a)
	{
		// d E / d lambda
		const double dE_dlambda = 1.0/(modeEigenValues[a]+modeRegularizer);

		// Dimension 3*n
		std::vector<double> dLambda_dXij;
		computeEigenValueDerivative(mode, modeDataSum, modeEigenVectors, a, m_iIndex, m_jIndex, dLambda_dXij);

		const double factor = modeWeight*dE_dlambda;
		
		const size_t eigenValueOffset = a*m_sampleDataDim;
		for(int b = 0; b < m_sampleDataDim; ++b)
		{
			tmpMemoryMatrix[eigenValueOffset+b] = factor*dLambda_dXij[b];
		}
	}

#pragma omp parallel for
	for(int b = 0; b < m_sampleDataDim; ++b)
	{
		for(int a = 0; a < numModeEigenValues; ++a)
		{
			const size_t eigenValueOffset = a*m_sampleDataDim;
			(*g)[b] += tmpMemoryMatrix[eigenValueOffset+b];
		}
	}
}

void TensorCompletionCostFunction::addCorruptShapeDataEnergy(double* f, vnl_vector<double>* g)
{
	if(m_corrDataWeight < math_eps || !m_bCurruptShape)
	{
		return;
	}

	//1. Reconstruct current shape
	//2. Transform to coordinate system of corrupt data
	//3. Compute nearest neighbors
	//4. Transform nearest neighbors to model coordinate system 
	//5. Compute data energy and gradient

	//Transform to coordinate system of corrupt data
	std::vector<double> currShape;
	currShape.resize(m_sampleDataDim,0.0);

//TODO: Parallelize
	for(size_t i = 0; i < m_sampleDataDim; ++i)
	{
		currShape[i] = m_updateData[m_shapeOffset+i];
	}

	MathHelper::transformData(m_corrShape_s, m_corrShape_R, "N", m_corrShape_t, "+", currShape);

#ifdef CHECK_CORRESPONDENCE_ANGLE
	std::vector<double> vertexNormals;
	bool bCheckNormals = m_pMesh != NULL;
	if(bCheckNormals)
	{	
		m_pMesh->setVertexList(currShape);
		MathHelper::computeVertexNormals(*m_pMesh, vertexNormals);
	}
#endif

	double corrShape_s_inv = 1.0/m_corrShape_s;
	//std::vector<double> corrShape_R_inv;
	//std::vector<double> corrShape_t_inv;
	//MathHelper::invertTransformation(m_corrShape_s, m_corrShape_R, "N", m_corrShape_t, "+", corrShape_s_inv, corrShape_R_inv, corrShape_t_inv);

	//Compute nearest neighbors
	const double sqrThresh = std::pow(m_corrShape_s*m_maxCorrDataDist,2);

	size_t numCorrVertices(0);
	std::vector<double> diffVec;
	diffVec.resize(m_sampleDataDim,0.0);

	for(size_t iVertex = 0; iVertex < m_numSampleVertices; ++iVertex)
	{
		std::vector<double> currTrafoVertex;
		currTrafoVertex.push_back(currShape[3*iVertex]);
		currTrafoVertex.push_back(currShape[3*iVertex+1]);
		currTrafoVertex.push_back(currShape[3*iVertex+2]);

		int pointIndex(0);
		double sqrDist(0.0);
		m_pCorrShapeKDTree->getNearestPoint(currTrafoVertex, pointIndex, sqrDist);

#if defined(USE_POINT_PLANE_CORRESPONDENCE) || defined(CHECK_CORRESPONDENCE_ANGLE)
		const Vec3d nnNormal((*m_pCorrShapeNormals)[3*pointIndex], (*m_pCorrShapeNormals)[3*pointIndex+1], (*m_pCorrShapeNormals)[3*pointIndex+2]);
#endif

#ifdef CHECK_CORRESPONDENCE_ANGLE
		if(bCheckNormals)
		{	
			const Vec3d sourceNormal(vertexNormals[3*iVertex], vertexNormals[3*iVertex+1], vertexNormals[3*iVertex+2]);
			const double angle = sourceNormal.angle(nnNormal);
			if(angle > MAX_ANGLE)
			{
				continue;
			}
		}
#endif

		std::vector<double> nnDiffVec;

#ifdef NN_DIST_THRESHOLD_METHOD
		//Method 1: Ignore vertices with distance to nearest neighbor larger than threshold
		if(sqrDist >= sqrThresh)
		{
			continue;
		}

#ifdef USE_POINT_PLANE_CORRESPONDENCE
		//Use point-plane correspondence
		const Vec3d sourcePoint(currShape[3*iVertex], currShape[3*iVertex+1], currShape[3*iVertex+2]);
		const Vec3d nnPoint((*m_pCorrShapeData)[3*pointIndex], (*m_pCorrShapeData)[3*pointIndex+1], (*m_pCorrShapeData)[3*pointIndex+2]);

		Vec3d planeProjectionPoint;					
		MathHelper::getPlaneProjection(sourcePoint, nnPoint, nnNormal, planeProjectionPoint);

		nnDiffVec.push_back(currTrafoVertex[0] - planeProjectionPoint[0]);
		nnDiffVec.push_back(currTrafoVertex[1] - planeProjectionPoint[1]);
		nnDiffVec.push_back(currTrafoVertex[2] - planeProjectionPoint[2]);
#else
		nnDiffVec.push_back(currTrafoVertex[0] - (*m_pCorrShapeData)[3*pointIndex]);
		nnDiffVec.push_back(currTrafoVertex[1] - (*m_pCorrShapeData)[3*pointIndex+1]);
		nnDiffVec.push_back(currTrafoVertex[2] - (*m_pCorrShapeData)[3*pointIndex+2]);
#endif

		// diffVec_model = 1/s*R^T*diffVec_target
		MathHelper::rotateData(m_corrShape_R,"T",nnDiffVec);
		MathHelper::scaleData(corrShape_s_inv,nnDiffVec);
#else
		//Method 2: Scale distance vectors to maximum threshold length if distance to nearest neighbor is larget than threshold
#ifdef USE_POINT_PLANE_CORRESPONDENCE
		//Use point-plane correspondence
		const Vec3d sourcePoint(currShape[3*iVertex], currShape[3*iVertex+1], currShape[3*iVertex+2]);
		const Vec3d nnPoint((*m_pCorrShapeData)[3*pointIndex], (*m_pCorrShapeData)[3*pointIndex+1], (*m_pCorrShapeData)[3*pointIndex+2]);

		Vec3d planeProjectionPoint;					
		MathHelper::getPlaneProjection(sourcePoint, nnPoint, nnNormal, planeProjectionPoint);

		nnDiffVec.push_back(currTrafoVertex[0] - planeProjectionPoint[0]);
		nnDiffVec.push_back(currTrafoVertex[1] - planeProjectionPoint[1]);
		nnDiffVec.push_back(currTrafoVertex[2] - planeProjectionPoint[2]);
#else
		nnDiffVec.push_back(currTrafoVertex[0] - (*m_pCorrShapeData)[3*pointIndex]);
		nnDiffVec.push_back(currTrafoVertex[1] - (*m_pCorrShapeData)[3*pointIndex+1]);
		nnDiffVec.push_back(currTrafoVertex[2] - (*m_pCorrShapeData)[3*pointIndex+2]);
#endif

		// diffVec_model = 1/s*R^T*diffVec_target
		MathHelper::rotateData(m_corrShape_R,"T",nnDiffVec);
		MathHelper::scaleData(corrShape_s_inv,nnDiffVec);

		if(sqrDist > sqrThresh)
		{			
			const double scaleFactor = sqrt(sqrThresh/sqrDist);
			nnDiffVec[0] *= scaleFactor;
			nnDiffVec[1] *= scaleFactor;
			nnDiffVec[2] *= scaleFactor;
		}
#endif

		diffVec[3*iVertex] = nnDiffVec[0];
		diffVec[3*iVertex+1] = nnDiffVec[1];
		diffVec[3*iVertex+2] = nnDiffVec[2];
		++numCorrVertices;
	}

	//Compute data energy and gradient
	if(numCorrVertices == 0)
	{
		std::cout << "No corresponding vertices of corrupt shape found" << std::endl;
		return;
	}

	const double factor = m_corrDataWeight/static_cast<double>(m_numSampleVertices);
	for(int b = 0; b < m_sampleDataDim; ++b)
	{
		(*f) += factor*std::pow(diffVec[b],2);
		(*g)[b] += 2.0*factor*diffVec[b];
	}
}

void TensorCompletionCostFunction::addShapeSmoothnessEnergy(double* f, vnl_vector<double>* g)
{
	if(m_precomputedSmoothnessIndices.empty() || m_smoothWeight < math_eps)
	{
		return;
	}

	const double regularizerFactor = m_smoothWeight / static_cast<double>(m_numSampleVertices);

	const size_t numVertices = m_precomputedSmoothnessIndices.size();
	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const std::vector<size_t>& currSmoothnessIndices = m_precomputedSmoothnessIndices[iVertex];
		const std::vector<double>& currSmoothnessWeights = m_precomputedSmoothnessWeights[iVertex];

		if(currSmoothnessIndices.empty())
		{
			continue;
		}

		double tmpX(0.0);
		double tmpY(0.0);
		double tmpZ(0.0);

		const size_t currNumSmoothnessIndices = currSmoothnessIndices.size();
		for(size_t j = 0; j < currNumSmoothnessIndices; ++j)
		{
			const size_t currVertexStartIndex = 3*currSmoothnessIndices[j];
			const size_t currSmoothStartIndex = m_shapeOffset + currVertexStartIndex;
			const double currSmoothWeight = currSmoothnessWeights[j];

			tmpX += currSmoothWeight*(m_updateData[currSmoothStartIndex]);
			tmpY += currSmoothWeight*(m_updateData[currSmoothStartIndex+1]);
			tmpZ += currSmoothWeight*(m_updateData[currSmoothStartIndex+2]);
		}

		if(!m_vertexLaplacian.empty())
		{
			tmpX -= m_vertexLaplacian[3*iVertex];
			tmpY -= m_vertexLaplacian[3*iVertex+1];
			tmpZ -= m_vertexLaplacian[3*iVertex+2];
		}

		(*f) += regularizerFactor*(std::pow(tmpX, 2) + std::pow(tmpY, 2) + std::pow(tmpZ, 2));

		for(size_t j = 0; j < currNumSmoothnessIndices; ++j)
		{
			//const size_t currVertexIndex = currSmoothnessIndices[j];
			const size_t currVertexStartIndex = 3*currSmoothnessIndices[j];

			const double currSmoothWeight = 2.0*regularizerFactor*currSmoothnessWeights[j];
			(*g)[currVertexStartIndex] += currSmoothWeight*tmpX;
			(*g)[currVertexStartIndex+1] += currSmoothWeight*tmpY;
			(*g)[currVertexStartIndex+2] += currSmoothWeight*tmpZ;
		}
	}
}

void TensorCompletionCostFunction::computeEigenValueDerivative(const size_t mode, const std::vector<double>& precomputedModeDataSum, const std::vector<double>& modeEigenVectors, const size_t eigenValueIndex
																					, const size_t iIndex, const size_t jIndex, std::vector<double>& dLambda_dXij)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
		return;
	}

	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;


	const bool bMode2 = (mode == 2);
	const size_t e2 = bMode2 ? d2 : d3;
	const size_t e3 = bMode2 ? d3 : d2;

	//Dimension 3n
	dLambda_dXij.resize(d1, 0.0);

	const size_t d1d2 = d1*d2;
	const size_t d2d3 = d2*d3;
	const size_t aEigenvectorIndex = eigenValueIndex*e2;

	// e_a[i]   i-th entry of a-th eigenvector
	// Mode2: ev_a_e = e_a[iIndex]
	// Mode3: ev_a_e = e_a[jIndex]
	const double ev_a_e = bMode2 ? modeEigenVectors[aEigenvectorIndex+iIndex] : modeEigenVectors[aEigenvectorIndex+jIndex];

	double ev_a_sum(0.0);
	for(size_t s = 0; s < e2; ++s)
	{
		// e_a[s]   r-th entry of a-th eigenvector
		const double ev_a_s = modeEigenVectors[aEigenvectorIndex+s];
		ev_a_sum += ev_a_s;
	}

	for(int r = 0; r < e2; ++r)
	{
		// e_a[r]   r-th entry of a-th eigenvector
		const double ev_a_r = modeEigenVectors[aEigenvectorIndex+r];

		const size_t r2i3 = bMode2 ? r : iIndex;
		const size_t j2r3 = bMode2 ? jIndex : r;

		//Mode 2: j*d2 + r 
		//Mode 3: r*d2 + i
		const size_t iShape = j2r3*d2 + r2i3;
		const size_t shapeIndex = m_semCorr[iShape];
		const size_t shapeOffset = shapeIndex*d1;

		const double d2d3ev_a_e = d2d3*ev_a_e;

#pragma omp parallel for
		for(int k = 0; k < d1; ++k)
		{
			//Mode2: e_ad2*d3
			const double currValue = d2d3ev_a_e*m_centeredData[shapeOffset+k]-ev_a_sum*precomputedModeDataSum[r*d1+k];
			dLambda_dXij[k] += ev_a_r*currValue;
		}
	}
	
	const double factor = 2.0/(d2d3*e3);

#pragma omp parallel for		
	for(int k = 0; k < m_sampleDataDim; ++k)
	{
		dLambda_dXij[k] *= factor;
	}
}

void TensorCompletionCostFunction::computeModeDataSum(const size_t mode, std::vector<double>& modeDataSum)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "TensorCompletionCostFunction::computeModeDataSum(...) - wrong mode input " << mode << std::endl;
		return;
	}

	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;

	modeDataSum.clear();

	bool bMode2 = (mode == 2);

	const size_t e2 = bMode2 ? d2 : d3;
	modeDataSum.resize(e2*d1, 0.0);

	if(mode == 2)
	{
		//Computes sum over all expressions (x_r = sum_{m=1}^{d3} c_{rm})
#pragma omp parallel for	
		for(int r = 0; r < d2; ++r)
		{
			const size_t rd1 = r*d1;
			for(int m = 0; m < d3; ++m)
			{
				const size_t iShape = m*d2+r;
				const size_t shapeIndex = m_semCorr[iShape];
				const size_t shapeOffset = shapeIndex*d1;

				for(int k = 0; k < d1; ++k)
				{
					modeDataSum[rd1+k] += m_centeredData[shapeOffset+k];
				}
			}
		}
	}
	else
	{
		//Computes sum over all identities (x_r = sum_{m=1}^{d2} c_{mr})
#pragma omp parallel for	
		for(int r = 0; r < d3; ++r)
		{
			const size_t rd1 = r*d1;
			for(int m = 0; m < d2; ++m)
			{
				const size_t iShape = r*d2+m;
				const size_t shapeIndex = m_semCorr[iShape];
				const size_t shapeOffset = shapeIndex*d1;

				for(int k = 0; k < d1; ++k)
				{
					modeDataSum[rd1+k] += m_centeredData[shapeOffset+k];
				}
			}
		}
	}
}

void TensorCompletionCostFunction::computeCovarianceEigenVectors(const size_t mode, std::vector<double>& eigenVectors, std::vector<double>& eigenValues)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "MDLShapeCostFunction::computeCovarianceEigenVectors(...) - wrong mode input " << mode << std::endl;
		return;
	}

	std::vector<size_t> modeDim;
	modeDim.push_back(m_sampleDataDim);
	modeDim.push_back(m_numIdentities);
	modeDim.push_back(m_numExpressions);


	size_t dim(0);
	std::vector<double> covarianceMatrix;
	if(!MathHelper::computeCovarianceMatrix(m_centeredData, m_semCorr, modeDim, mode, covarianceMatrix, dim))
	{
		std::cout << "Failed computing covariance matrix" << std::endl;
		return;
	}

	std::vector<double> tmpEigenVectors;
	std::vector<double> tmpEigenValues;
	MathHelper::computeLeftSingularVectors(covarianceMatrix, dim, dim, tmpEigenVectors, tmpEigenValues);

	size_t numEigenValues(0);
	for(size_t i = 0; i < tmpEigenValues.size(); ++i)
	{
		if(fabs(tmpEigenValues[i]) > math_eps)
		{
			++numEigenValues;
		}
	}

	eigenVectors.resize(dim*numEigenValues);
	eigenValues.resize(numEigenValues);

#pragma omp parallel for
	for(int i = 0; i < numEigenValues; ++i)
	{
		const size_t startIndex = i*dim;
		for(size_t j = 0; j < dim; ++j)
		{
			const size_t index = startIndex+j;
			eigenVectors[index] = tmpEigenVectors[index];
		}

		eigenValues[i] = tmpEigenValues[i];
	}

#ifdef OUTPUT_MODE_COMPACTNESS
	double sum(0.0);
	std::cout << "EigenValues: ";
	for(size_t i = 0; i < tmpEigenValues.size(); ++i)
	{
		std::cout << tmpEigenValues[i] << " ";
		sum += tmpEigenValues[i];
	}

	std::cout << std::endl;
	std::cout << "Compactness mode " << mode << std::endl;
	
	double tmpSum(0.0);
	for(size_t i = 0; i < tmpEigenValues.size(); ++i)
	{
		tmpSum += tmpEigenValues[i];
		const double currValue = 100.0*tmpSum/sum;
		std::cout << std::setprecision(3) <<  tmpSum << " (" << currValue << ") " << std::endl;
	}
#endif
}

bool TensorCompletionCostFunction::checkCurruptShapeData()
{
	if(m_pCorrShapeData == NULL
		|| m_pCorrShapeNormals == NULL
		|| m_pCorrShapeKDTree == NULL)
	{
		std::cout << "Not initialized data" << std::endl;
		return false;
	}

	if(m_corrShape_R.size() != 9
		|| m_corrShape_t.size() != 3)
	{
		std::cout << "Transformation of wrong dimension" << std::endl;
		return false;
	}

	return true;
}