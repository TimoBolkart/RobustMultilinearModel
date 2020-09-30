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

#ifndef MDLHELPER_H
#define MDLHELPER_H

#include "DataContainer.h"
#include "KDTree3.h"

#include <stdlib.h>
#include <vector>
#include <string>

#include <vnl/vnl_cost_function.h>
#include <vnl/algo/vnl_lbfgsb.h>

class MDLHelper
{
public:
	static bool computeThinPlateSplines(const std::string& sstrFileCollectionName, const std::string& sstrTextureCoordsFileName, const std::string& sstrOutFolder);

private:
	static void updateShapeParameter(const size_t iShape, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation);

	static void updateParameter(const size_t numSamples, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation);

	static void updateShapeData(const size_t iShape, const std::vector<std::vector<double>>& vecCs, const std::vector<std::vector<double>>& matAs, const std::vector<std::vector<double>>& matWs, const std::vector<std::vector<double>>& sourcePointsVec
										, const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data);

	static void updateData(std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
								  ,const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data);

	static void computeShapeExcludedMean(const std::vector<double>& data, const size_t shapeDim, const size_t iExcludedShape, std::vector<double>& excludedShapeMean);

	static bool alignShapeData(const size_t iShape, const std::vector<double>& target, std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs);

	static bool procrustesAlignShapeData(std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, const size_t numIter);

public:
	static void outputParameterVariations(const size_t numSamples, const size_t numVertices, const std::vector<double>& parameterVariation, const std::string& sstrOutFolder, const std::string& sstrFileName);

	static void outputShapeData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames, const size_t iSample);

	static void outputData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames);

	static void precomputeBiLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights);

	static void precomputeLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights);

	static void precomputeVertexLaplacians(const std::vector<double>& data, const size_t numShapes, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices
														, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, std::vector<std::vector<double>>& vertexLaplacians);

	static void MDLHelper::precomputeVertexLaplacian(const std::vector<double>& data, const size_t iShape, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights
																	, std::vector<double>& vertexLaplacian);



	static void robustModelLearning(const std::string& sstrFileCollectionName, const std::string& sstrLmkIndexFileName, const std::string & sstCorruptFileCollectionName 
											, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName, const std::string& sstrOutFolder);

private:
	static bool getRMMData(const std::string& sstrFileCollectionName, const std::string& sstrModelLmkIndexFileName, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName, const std::string & sstCorruptFileCollectionName
							, std::vector<double>& data, std::vector<std::string>& geometryFileNames, size_t& numIdentities, size_t& numExpressions, DataContainer& mesh, std::vector<size_t>& modelLmkIndices
							, std::vector<bool>& missingShapes, std::vector<bool>& corruptShapes, std::vector<std::vector<double>>& corruptData, std::vector<std::vector<std::pair<double,bool>>>& curruptDataLmks, std::vector<std::vector<double>>& corruptDataNormals
							, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
							, std::vector<size_t>& outerBoundaryVertexIDs, std::vector<size_t>& innerBoundaryVertexIDs);

	static void optimizeMissingShape(std::vector<double>& data, const std::vector<size_t>& modeDim, const std::vector<size_t>& semCorr, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes
												, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const std::vector<double>& vertexLaplacians, const vnl_vector<long>& boundSelection
												, const vnl_vector<double>& lowerShapeBounds, const vnl_vector<double>& upperShapeBounds, const size_t iShape);

	static void	optimizeCorruptShape(std::vector<double>& data, const std::vector<size_t>& modeDim, const std::vector<size_t>& semCorr, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const std::vector<double>& corruptData
											, const std::vector<double>& corruptDataNormals, const KDTree3* pCorrShapesKDTree, const double maxCorrDataDist, const double s, const std::vector<double>& R, const std::vector<double>& t
											, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const std::vector<double>& vertexLaplacians
											, const vnl_vector<long>& boundSelection, const vnl_vector<double>& lowerShapeBounds, const vnl_vector<double>& upperShapeBounds, const size_t iShape);

	static void optimizeCorrespondence(std::vector<double>& data, const std::vector<double>& initialParam, std::vector<double>& parameterVariation, const std::vector<size_t>& modeDim, const std::vector<size_t>& semCorr
													, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
													, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights
													, const vnl_vector<long>& boundSelection, const vnl_vector<double>& lowerShapeBounds, const vnl_vector<double>& upperShapeBounds, const size_t iShape);



	static void convertParameters(const std::vector<std::vector<double>>& shapeParam, const size_t numShapes, const size_t shapeParameterDim, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, std::vector<double>& param);

	static bool computeRMMOptimizationBounds(const size_t numShapes, const size_t shapeDataDim, const size_t shapeParameterDim, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const std::vector<double>& initialParam
														, const std::vector<size_t>& outerBoundaryVertexIDs, const std::vector<size_t>& innerBoundaryVertexIDs
														, vnl_vector<long>& completionBoundSelection, vnl_vector<long>& optimizationBoundSelection, std::vector<vnl_vector<double>>& lowerShapeBounds, std::vector<vnl_vector<double>>& upperShapeBounds);

	static void computeKDTrees(const size_t numShapes, const std::vector<bool>& corruptShapes, const std::vector<std::vector<double>>& corruptData, std::vector<KDTree3*>& corrShapesKDTrees);

	static bool computeCorruptDataLmkAlignment(const std::vector<double>& data, const size_t numShapes, const size_t shapeDataDim, const std::vector<bool>& corruptShapes, const std::vector<size_t>& modelLmkIndices
														, const std::vector<std::vector<std::pair<double,bool>>>& curruptDataLmks, std::vector<double>& corrShapes_s, std::vector<std::vector<double>>& corrShapes_R, std::vector<std::vector<double>>& corrShapes_t);

	static void initializeMissingData(std::vector<double>& data, const std::vector<size_t>& modeDim, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes);

	static void getBoundaryNeighbors(const DataContainer& mesh, const std::vector<size_t>& boundary, std::vector<size_t>& neighbors);

	static void outputConfig(const std::string& sstrOutFolder);

public:
	static void outputCompactness(const std::vector<double>& data, const std::vector<size_t>& semCorr, const size_t d1, const size_t d2, const size_t d3, const std::string& sstrOutFileName);

private:
	static std::string getConfigOutFolder(const std::string& sstrOutFolder);
};

#endif