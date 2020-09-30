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

#include "MDLShapeCostFunction.h"
#include "MathHelper.h"
#include "Definitions.h"

#include <math.h>
#include <iostream>

//#define OUTPUT_MODE_COMPACTNESS
#ifdef OUTPUT_MODE_COMPACTNESS
#include <iomanip>  
#endif

//#define OUTPUT_FUNCTION_VALUE
//#define OUTPUT_TIME

//#undef DEBUG_OUTPUT

#ifdef OUTPUT_TIME
#include "PerformanceCounter.h"
#endif

const double MODE_2_REGULARIZER = 0.01;
const double MODE_3_REGULARIZER = 0.01;

MDLShapeCostFunction::MDLShapeCostFunction(std::vector<double>& data, const std::vector<double>& initialParam, const size_t numIdentities, const size_t numExpressions, const std::vector<std::vector<double>>& vecCs, const std::vector<std::vector<double>>& matAs
											, const std::vector<std::vector<double>>& matWs, const std::vector<std::vector<double>>& sourcePointsVec, const std::vector<size_t>& semCorr, const double identityWeight, const double expressionWeight
											, const double maxParameterVariation, const size_t iShape)
: vnl_cost_function(2*data.size()/(3*numIdentities*numExpressions))
, m_data(data)
, m_initialParam(initialParam)
, m_dataSize(data.size())
, m_numIdentities(numIdentities)
, m_numExpressions(numExpressions)
, m_numSamples(numIdentities*numExpressions)
, m_sampleDataDim(m_dataSize/m_numSamples)
, m_sampleParamDim(2*m_sampleDataDim/3)
, m_numSampleVertices(m_sampleDataDim/3)
, m_vecCs(vecCs)
, m_matAs(matAs)
, m_matWs(matWs)
, m_sourcePointsVec(sourcePointsVec)
, m_semCorr(semCorr)
, m_identityWeight(identityWeight)
, m_expressionWeight(expressionWeight)
, m_lmkWeight(0.0)
, m_smoothWeight(0.0)
, m_maxSqrParameterVariation(std::pow(maxParameterVariation, 2))
, m_bUpdateData(false)
, m_iIndex(iShape%numIdentities)
, m_jIndex((iShape-m_iIndex)/numIdentities)
#ifdef OUTPUT_GRADIENT_NORM
, m_gradientNorm(0.0)
#endif
{
	MathHelper::centerData(m_data, m_sampleDataDim, m_centeredData, m_dataMean);

	m_dataParam_ij.resize(m_sampleParamDim, 0.0);

	const size_t shapeIndex = m_semCorr[iShape];
	const size_t dataOffset = shapeIndex*m_sampleParamDim;
	for(size_t i = 0; i < m_sampleParamDim; ++i)
	{
		m_dataParam_ij[i] = m_initialParam[dataOffset+i];
	}
}

MDLShapeCostFunction::~MDLShapeCostFunction()
{

}

bool MDLShapeCostFunction::setLandmarks(const std::vector<size_t>& lmkIndices, std::vector<double>& lmks, const double lmkWeight)
{
	const size_t numLmks = lmkIndices.size();
	if(lmks.size() != 3*numLmks*m_numSamples)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Unable to set landmarks due to wrong dimension " << lmks.size() << " != " << numLmks*m_numSamples << std::endl;
#endif

		return false;
	}

	m_lmkIndices = lmkIndices;
	m_lmks = lmks;
	m_numLmks = numLmks;
	m_lmkWeight = lmkWeight;

	return true;
}

bool MDLShapeCostFunction::setSmoothnessValues(const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const double smoothnessWeight)
{
	if(precomputedSmoothnessIndices.size() != precomputedSmoothnessWeights.size())
	{
		return false;
	}

	m_precomputedSmoothnessIndices = precomputedSmoothnessIndices;
	m_precomputedSmoothnessWeights = precomputedSmoothnessWeights;
	m_smoothWeight = smoothnessWeight;
	return true;
}

void MDLShapeCostFunction::compute(const vnl_vector<double>& x, double *f, vnl_vector<double>* g)
{
#ifdef OUTPUT_TIME
	const double t1 = PerformanceCounter::getTime();
#endif

	updateShapeParametrization(x, m_iIndex, m_jIndex);

#ifdef OUTPUT_TIME
	const double t2 = PerformanceCounter::getTime();
#endif
	
	if(m_bUpdateData)
	{
		updateShapeData(m_iIndex, m_jIndex);

		MathHelper::centerData(m_data, m_sampleDataDim, m_centeredData, m_dataMean);
		m_bUpdateData = false;
	}

#ifdef OUTPUT_TIME
	const double t3 = PerformanceCounter::getTime();
#endif

	init(f, g);

#ifdef OUTPUT_TIME
	const double t4 = PerformanceCounter::getTime();
#endif

	std::vector<double> mode2EigenVectors; 
	std::vector<double> mode2EigenValues;
	computeCovarianceEigenVectors(2, mode2EigenVectors, mode2EigenValues);
	
#ifdef OUTPUT_TIME
	const double t5 = PerformanceCounter::getTime();
#endif

	std::vector<double> mode3EigenVectors; 
	std::vector<double> mode3EigenValues;
	computeCovarianceEigenVectors(3, mode3EigenVectors, mode3EigenValues);

#ifdef OUTPUT_TIME
	const double t6 = PerformanceCounter::getTime();
#endif

	addModeCompactnessEnergy(2, mode2EigenValues, f);

#ifdef OUTPUT_TIME
	const double t7 = PerformanceCounter::getTime();
#endif

	addModeCompactnessEnergy(3, mode3EigenValues, f);

#ifdef OUTPUT_TIME
	const double t8 = PerformanceCounter::getTime();
#endif


#ifdef OUTPUT_FUNCTION_VALUE
	const double compactnessValue = *f;
#endif

	std::vector<double> dXij_dalpha;
	computeShapeParameterDerivative(m_iIndex, m_jIndex, dXij_dalpha);

#ifdef OUTPUT_TIME
	const double t9 = PerformanceCounter::getTime();
#endif

	addModeShapeCompactnessGradient(2, mode2EigenVectors, mode2EigenValues, m_iIndex, m_jIndex, dXij_dalpha, g);

#ifdef OUTPUT_TIME
	const double t10 = PerformanceCounter::getTime();
#endif

	addModeShapeCompactnessGradient(3, mode3EigenVectors, mode3EigenValues, m_iIndex, m_jIndex, dXij_dalpha, g);

#ifdef OUTPUT_TIME
	const double t11 = PerformanceCounter::getTime();
#endif
	
	addShapeLandmarkEnergy(m_iIndex, m_jIndex, dXij_dalpha, f, g);

#ifdef OUTPUT_TIME
	const double t12 = PerformanceCounter::getTime();
#endif

#ifdef OUTPUT_FUNCTION_VALUE
	const double landmarkValue = *f-compactnessValue;
#endif

	addShapeSmoothnessEnergy(m_iIndex, m_jIndex, dXij_dalpha, f, g);

#ifdef OUTPUT_TIME
	const double t13 = PerformanceCounter::getTime();
#endif

#ifdef OUTPUT_GRADIENT_NORM
	m_gradientNorm = computeGradientNorm(g);
#endif

#ifdef OUTPUT_FUNCTION_VALUE
	std::cout << "Function evaluation" << std::endl;
	std::cout << "Compactness energy: " << compactnessValue << std::endl;
	std::cout << "Landmark energy: " << landmarkValue << std::endl;
	std::cout << "Smoothness energy: " << *f-landmarkValue-compactnessValue << std::endl;
	std::cout << "Function value: " << *f << std::endl;
#ifdef OUTPUT_GRADIENT_NORM
	std::cout << "Gradient norm: " << m_gradientNorm << std::endl;
#endif
	std::cout << std::endl;
#endif

#ifdef OUTPUT_TIME
	const double paramUpdateDiff = t2-t1; 
	const double dataUpdateDiff = t3-t2; 
	const double initParamDiff = t4-t3; 
	const double mode2EVsDiff = t5-t4; 
	const double mode3EvsDiff = t6-t5; 
	const double mode2CompEnergyDiff = t7-t6; 
	const double mode3CompEnergyDiff = t8-t7; 
	const double shapeParamDerivDiff = t9-t8; 
	const double mode2CompGradDiff = t10-t9; 
	const double mode3CompGradDiff = t11-t10; 
	const double lmkEnergyDiff = t12-t11; 
	const double smoothnessEnergyDiff = t13-t12; 
	const double overallTime = t13-t1; 

	std::cout << std::endl;
	std::cout << "Function evaluation timing: " << std::endl;
	std::cout << "Parameter update: " << paramUpdateDiff << " s  " << paramUpdateDiff/60 << " min" << std::endl;
	std::cout << "Data update: " << dataUpdateDiff << " s  " << dataUpdateDiff/60 << " min" << std::endl;
	std::cout << "Parameter init: " << initParamDiff << " s  " << initParamDiff/60 << " min" << std::endl;
	std::cout << "Mode 2 EVs: " << mode2EVsDiff << " s  " << mode2EVsDiff/60 << " min" << std::endl;
	std::cout << "Mode 3 EVs: " << mode3EvsDiff << " s  " << mode3EvsDiff/60 << " min" << std::endl;
	std::cout << "Mode 2 Comp Energy: " << mode2CompEnergyDiff << " s  " << mode2CompEnergyDiff/60 << " min" << std::endl;
	std::cout << "Mode 3 Comp Energy: " << mode3CompEnergyDiff << " s  " << mode3CompEnergyDiff/60 << " min" << std::endl;
	std::cout << "Shape param deriv: " << shapeParamDerivDiff << " s  " << shapeParamDerivDiff/60 << " min" << std::endl;
	std::cout << "Mode 2 Comp Grad: " << mode2CompGradDiff << " s  " << mode2CompGradDiff/60 << " min" << std::endl;
	std::cout << "Mode 3 Comp Grad: " << mode3CompGradDiff << " s  " << mode3CompGradDiff/60 << " min" << std::endl;
	std::cout << "Lmk Energy: " << lmkEnergyDiff << " s  " << lmkEnergyDiff/60 << " min" << std::endl;
	std::cout << "Smooth Energy: " << smoothnessEnergyDiff << " s  " << smoothnessEnergyDiff/60 << " min" << std::endl;
	std::cout << "Overall time: " << overallTime << " s  " << overallTime/60 << " min" << std::endl;
	std::cout << std::endl;
#endif


	m_bUpdateData = true;
/**
	{
		std::fstream outStream("E:\\Computations\\MDL\\TestData\\TestOutput\\DataComputeEnd.txt", std::ios::out);

		for(size_t iSample = 0; iSample < m_numSamples; ++iSample)
		{
			const size_t dataOffset = iSample*m_sampleDataDim;

			for(size_t j = 0; j < m_sampleDataDim; ++j)
			{
				outStream << m_data[dataOffset+j] << " ";
			}
			outStream << std::endl;
		}

		outStream.close();
	}
**/
}

void MDLShapeCostFunction::init(double* f, vnl_vector<double>* g)
{
	*f = 0.0;

	const size_t dim = m_sampleParamDim;
		
	//TODO Run in parallel
	//for(size_t i = 0; i < dim; ++i)
#pragma omp parallel for
	for(int i = 0; i < dim; ++i)
	{
		(*g)[i] = 0.0;
	}
}

void MDLShapeCostFunction::addModeCompactnessEnergy(const size_t mode, const std::vector<double>& modeEigenValues, double* f)
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

void MDLShapeCostFunction::addModeShapeCompactnessGradient(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha
																			, vnl_vector<double>* g)
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
	tmpMemoryMatrix.resize(m_sampleParamDim * numModeEigenValues, 0.0);

	//Pre-compute the sum over all expressions (mode == 2) or identities (mode == 3)
	std::vector<double> modeDataSum;
	computeModeDataSum(mode, modeDataSum);

#pragma omp parallel for
	for(int a = 0; a < numModeEigenValues; ++a)
	{
		const size_t eigenValueOffset = a*m_sampleParamDim;

		// Dimension 3*n
		std::vector<double> dLambda_dXij;
		computeEigenValueDerivative(mode, modeDataSum, modeEigenVectors, a, iIndex, jIndex, dLambda_dXij);

		for(size_t b = 0; b < m_numSampleVertices; ++b)
		{
			const size_t tmp2b = 2*b;
			const size_t tmp3b = 3*b;

			// (dXij[v_b] / dalpha[v_b]) * (dLambda / dXij[v_b])
			const double val1 = dXij_dalpha[tmp2b]*dLambda_dXij[tmp3b] + dXij_dalpha[m_sampleParamDim+tmp2b]*dLambda_dXij[tmp3b+1] + dXij_dalpha[2*m_sampleParamDim+tmp2b]*dLambda_dXij[tmp3b+2];
			const double val2 = dXij_dalpha[tmp2b+1]*dLambda_dXij[tmp3b] + dXij_dalpha[m_sampleParamDim+tmp2b+1]*dLambda_dXij[tmp3b+1] + dXij_dalpha[2*m_sampleParamDim+tmp2b+1]*dLambda_dXij[tmp3b+2];

			tmpMemoryMatrix[eigenValueOffset+tmp2b] = val1;
			tmpMemoryMatrix[eigenValueOffset+tmp2b+1] = val2;
		}
	}

#pragma omp parallel for
	for(int b = 0; b < m_sampleParamDim; ++b)
	{
		for(int a = 0; a < numModeEigenValues; ++a)
		{
			const size_t eigenValueOffset = a*m_sampleParamDim;

			// d E / d lambda
			const double dE_dlambda = 1.0/(modeEigenValues[a]+modeRegularizer);
			const double factor = modeWeight*dE_dlambda;
		
			(*g)[b] += factor*tmpMemoryMatrix[eigenValueOffset+b];
		}
	}
}

void MDLShapeCostFunction::addShapeLandmarkEnergy(const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha, double* f, vnl_vector<double>* g)
{
	if(m_numLmks == 0 || m_lmkWeight < math_eps)
	{
		return;
	}

	//const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;

	const double regularizerFactor = m_lmkWeight / static_cast<double>(m_numLmks*d2*d3);

	const size_t iShape = (jIndex*d2 + iIndex);
	const size_t shapeIndex = m_semCorr[iShape];

	const size_t shapeOffset = shapeIndex*m_sampleDataDim;
	const size_t lmkOffset = shapeIndex*3*m_numLmks;

	double value(0.0);

	for(size_t iLmk = 0; iLmk < m_numLmks; ++iLmk)
	{
		const size_t currLmkIndex = m_lmkIndices[iLmk];

		const size_t tmp2currLmkIndex = 2*currLmkIndex;
		const size_t currLmkDataOffset = shapeOffset+3*currLmkIndex;
		const size_t currLmkOffset = lmkOffset+3*iLmk;

		const double tmpX = m_data[currLmkDataOffset]-m_lmks[currLmkOffset];
		const double tmpY = m_data[currLmkDataOffset+1]-m_lmks[currLmkOffset+1];
		const double tmpZ = m_data[currLmkDataOffset+2]-m_lmks[currLmkOffset+2];
		value += pow(tmpX, 2) + pow(tmpY, 2), + pow(tmpZ, 2);

		const double val1 = dXij_dalpha[tmp2currLmkIndex]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currLmkIndex]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currLmkIndex]*tmpZ;
		const double val2 = dXij_dalpha[tmp2currLmkIndex+1]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currLmkIndex+1]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currLmkIndex+1]*tmpZ;

		(*g)[tmp2currLmkIndex] += 2.0*regularizerFactor*val1;
		(*g)[tmp2currLmkIndex+1] += 2.0*regularizerFactor*val2;
	}

	value *= regularizerFactor; 
	(*f) += value;
}

void MDLShapeCostFunction::addShapeSmoothnessEnergy(const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha, double* f, vnl_vector<double>* g)
{
	if(m_precomputedSmoothnessIndices.empty() || m_smoothWeight < math_eps)
	{
		return;
	}

	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	//const size_t d3 = m_numExpressions;

	//Since we only optimize for one shape and therefore only the smoothness of current shape is considered for optimization,
	//there is no regularization w.r.t. d2 and d3
	//const double regularizerFactor = m_smoothWeight / static_cast<double>(m_numSampleVertices*d2*d3);
	const double regularizerFactor = m_smoothWeight / static_cast<double>(m_numSampleVertices);

	const size_t iShape = jIndex*d2 + iIndex;
	const size_t shapeIndex = m_semCorr[iShape];
	const size_t shapeOffset = shapeIndex*d1;

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
			const size_t currSmoothStartIndex = shapeOffset + currVertexStartIndex;
			const double currSmoothWeight = currSmoothnessWeights[j];

			tmpX += currSmoothWeight*(m_data[currSmoothStartIndex]);
			tmpY += currSmoothWeight*(m_data[currSmoothStartIndex+1]);
			tmpZ += currSmoothWeight*(m_data[currSmoothStartIndex+2]);
		}

		(*f) += regularizerFactor*(std::pow(tmpX, 2) + std::pow(tmpY, 2) + std::pow(tmpZ, 2));

		for(size_t j = 0; j < currNumSmoothnessIndices; ++j)
		{
			const size_t currVertexIndex = currSmoothnessIndices[j];

			const size_t tmp2currVertexIndex = 2*currVertexIndex;
								
			const double val1 = dXij_dalpha[tmp2currVertexIndex]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currVertexIndex]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currVertexIndex]*tmpZ;
			const double val2 = dXij_dalpha[tmp2currVertexIndex+1]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currVertexIndex+1]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currVertexIndex+1]*tmpZ;

			const double currSmoothWeight = 2.0*regularizerFactor*currSmoothnessWeights[j];
			(*g)[tmp2currVertexIndex] += currSmoothWeight*val1;
			(*g)[tmp2currVertexIndex+1] += currSmoothWeight*val2;
		}
	}
}

void MDLShapeCostFunction::computeEigenValueDerivative(const size_t mode, const std::vector<double>& precomputedModeDataSum, const std::vector<double>& modeEigenVectors, const size_t eigenValueIndex
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

void MDLShapeCostFunction::computeShapeParameterDerivative(const size_t iIndex, const size_t jIndex, std::vector<double>& dXij_dalpha)
{
	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;

	//Dimension 2n x 3
	dXij_dalpha.resize(d1*m_sampleParamDim, 0.0);

	const size_t ijIndex = jIndex*d2+iIndex;
	const size_t ijShapeIndex = m_semCorr[ijIndex];
	const std::vector<double>& matAij = m_matAs[ijShapeIndex];
	const std::vector<double>& matWij = m_matWs[ijShapeIndex];
	const std::vector<double>& currSourcePoints = m_sourcePointsVec[ijShapeIndex];

	const double sqr_math_eps = std::pow(math_eps, 2);

	for(size_t k = 0; k < m_numSampleVertices; ++k)
	{
		// d sigma / d alpha 
		//Dimension 2 x n
		std::vector<double> tmpDerivMatrix;
		tmpDerivMatrix.resize(2*m_numSampleVertices, 0.0);

		const double alphaij_xk_1 = m_dataParam_ij[2*k];
		const double alphaij_xk_2 = m_dataParam_ij[2*k+1];

	#pragma omp parallel for
		for(int i = 0; i < m_numSampleVertices; ++i)
		{
			const size_t tmp2i = 2*i;

			const double uDiff = alphaij_xk_1 - currSourcePoints[2*i];
			const double vDiff = alphaij_xk_2 - currSourcePoints[2*i+1];
			const double tmpSqrLength = std::pow(uDiff, 2) + std::pow(vDiff, 2);
			if(tmpSqrLength > sqr_math_eps)
			{
				const double factor = 2.0*log(sqrt(tmpSqrLength))+1.0;
				tmpDerivMatrix[tmp2i] = factor*uDiff;
				tmpDerivMatrix[tmp2i+1] = factor*vDiff;
			}
		}

		//Dimension 2 x 3
		std::vector<double> outMat;
		MathHelper::matrixMult(tmpDerivMatrix, 2, m_numSampleVertices, "N", matWij, 3, outMat);

		//dXij_dalpha[v_k] / dalpha[v_k] = A_ij^T + tmpDerivMatrix*W_ij
		const size_t tmp2k = 2*k;

	//#pragma omp parallel for
		for(int col = 0; col < 3; ++col)
		{
			const size_t col2 = col*2;
			const size_t col2n = col*m_sampleParamDim;
			for(size_t row = 0; row < 2; ++row)
			{
				dXij_dalpha[col2n+tmp2k+row] = matAij[row*3+col] + outMat[col2+row];
			}
		}
	}
}

void MDLShapeCostFunction::updateShapeParametrization(const vnl_vector<double>& x, const size_t iIndex, const size_t jIndex)
{
	const size_t d2 = m_numIdentities;

	const size_t iShape = jIndex*d2+iIndex;
	const size_t shapeIndex = m_semCorr[iShape];
	const size_t paramOffset = shapeIndex*m_sampleParamDim;

#pragma omp parallel for
	for(int iVertex = 0; iVertex < m_numSampleVertices; ++iVertex)
	{
		const size_t tmpParamStartIndex = 2*iVertex;

		double deltaAlpha1 = x[tmpParamStartIndex];
		double deltaAlpha2 = x[tmpParamStartIndex+1];
		const double tmpSqrLength = std::pow(deltaAlpha1,2) + std::pow(deltaAlpha2, 2);
		if(tmpSqrLength > m_maxSqrParameterVariation)
		{
			const double factor = sqrt(m_maxSqrParameterVariation/tmpSqrLength);
			deltaAlpha1 *= factor;
			deltaAlpha2 *= factor;
		}

		const double initAlpha1 = m_initialParam[paramOffset+tmpParamStartIndex];
		const double initAlpha2 = m_initialParam[paramOffset+tmpParamStartIndex+1];

		m_dataParam_ij[tmpParamStartIndex] = initAlpha1+deltaAlpha1;
		m_dataParam_ij[tmpParamStartIndex+1] = initAlpha2+deltaAlpha2; 
	}
}

void MDLShapeCostFunction::updateShapeData(const size_t iIndex, const size_t jIndex)
{
	//const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;

	const size_t iShape = jIndex*d2+iIndex;
	const size_t shapeIndex = m_semCorr[iShape];
	const size_t dataOffset = shapeIndex*m_sampleDataDim;

	const std::vector<double>& currVecC = m_vecCs[shapeIndex]; 
	const std::vector<double>& currMatA = m_matAs[shapeIndex];
	const std::vector<double>& currMatW = m_matWs[shapeIndex];
	const std::vector<double>& currSourcePoints = m_sourcePointsVec[shapeIndex];

	//TODO Run in parallel
	//for(size_t iVertex = 0; iVertex < m_numSampleVertices; ++iVertex)
#pragma omp parallel for
	for(int iVertex = 0; iVertex < m_numSampleVertices; ++iVertex)
	{
		const size_t tmpDataStartIndex = 3*iVertex;
		const size_t tmpParamStartIndex = 2*iVertex;
		
		std::vector<double> paramVertex;
		paramVertex.push_back(m_dataParam_ij[tmpParamStartIndex]);
		paramVertex.push_back(m_dataParam_ij[tmpParamStartIndex+1]);

		std::vector<double> outDataVertex;
		if(!MathHelper::evaluateInterpolation(currVecC, currMatA, currMatW, currSourcePoints, paramVertex, outDataVertex))
		{
			std::cout << "MDLShapeCostFunction::updateData(...) - unable to compute tps point" << std::endl;
			continue;
		}

		m_data[dataOffset+tmpDataStartIndex] = outDataVertex[0];
		m_data[dataOffset+tmpDataStartIndex+1] = outDataVertex[1];
		m_data[dataOffset+tmpDataStartIndex+2] = outDataVertex[2];
	}
}

void MDLShapeCostFunction::computeModeDataSum(const size_t mode, std::vector<double>& modeDataSum)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "MDLShapeCostFunction::computeModeDataSum(...) - wrong mode input " << mode << std::endl;
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

void MDLShapeCostFunction::computeCovarianceEigenVectors(const size_t mode, std::vector<double>& eigenVectors, std::vector<double>& eigenValues)
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

#ifdef OUTPUT_GRADIENT_NORM
double MDLShapeCostFunction::computeGradientNorm(const vnl_vector<double>* g) const
{
	double norm(0.0);

	for(size_t i = 0; i < m_sampleParamDim; ++i)
	{
		norm += std::pow((*g)[i], 2);
	}

	return sqrt(norm);
}
#endif

