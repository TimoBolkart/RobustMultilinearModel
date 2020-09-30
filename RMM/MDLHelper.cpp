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

#include "MDLHelper.h"
#include "RelabelMultilinearData.h"
#include "MDLShapeCostFunction.h"
#include "TensorCompletionCostFunction.h"
#include "FileLoader.h"
#include "FileWriter.h"
#include "MathHelper.h"

#include <iostream>
#include <fstream>

#include <vector>
#include <set>
#include <map>

#define OUTPUT_INITIAL_ALIGNED_DATA
#define OUTPUT_TIME
#define OUTPUT_COMPACTNESS
#define OUTPUT_ITERATION_RESULTS
//#define OUTPUT_MEAN_FACE

//If disabled, Laplacian smoothness is used instead
#define USE_BI_LAPLACIAN_SMOOTHNESS

#ifdef OUTPUT_COMPACTNESS
#include "MultilinearModel.h"
#endif

#ifdef OUTPUT_TIME
#include "PerformanceCounter.h"
#endif

#include <iomanip>  

//If enabled, semantic correspondence is optimized (time consuming)
#define OPTIMIZE_SEMANTIC_CORRESPONDENCE
//If enabled, the vertex correspondence is optimized (time consuming)
#define OPTIMIZE_VERTEX_CORRESPONDENCE

//Weight of identity and expression compactness energy
const double IDENTITY_WEIGHT = 1.0;
const double EXPRESSION_WEIGHT = 1.0;

//Weight of corrupt data energy
const double CORRUPT_DATA_WEIGHT = 0.002;
//Truncation threshold on the corrupt data distance to be robust to outliers
const double MAX_CORR_DATA_DIST = 5.0;

//Weight of the regularization energy 
//We choose w_R = 20.0 to complete missing and corrupt data, and w_R = 0.5 to optimize the vertex correspondence
//(We recommend using larger values for the vertex correspondence in the case of much smaller datasets)
const double DATA_COMPLETION_SMOOTHNESS_WEIGHT = 20.0;
const double CORRESPONDENCE_OPTIMIZATION_SMOOTHNESS_WEIGHT = 20.0;//0.5;

//Number of iterations where one iteration means optimizing over each shapes once
const size_t NUM_ITERATION = 10;
//Number of iterations during semantic correspondence optimization
const size_t NUM_SEM_CORR_OPT = 1;

//Number of function evaluations during optimization of one shape
const size_t NUM_FKT_EVAL = 50;
//Number of function evaluations before re-computing the shape alignment (should be fraction of NUM_FKT_EVAL)
const size_t NUM_NUM_FKT_EVAL_ALIGNMENT = 10;

//Maximum variation of a vertex during optimization from the initialization for missing and corrupt data completion
const double MAX_COORD_VARIATON = 10000.0;
//Maximum variation of a vertex in parameter space during optimization from the initialization for correspondence optimization
const double MAX_PARAMETER_VARIATION = 1.0;

const double OPTIMIZATION_DOMAIN_MIN = -0.2;
const double OPTIMIZATION_DOMAIN_MAX = 1.0-OPTIMIZATION_DOMAIN_MIN;

//Boundary vertices are allowed to vary within [p-delta, p+delta]
//If value is zero, boundary vertex is fixed during optimization
const double MAX_OUTER_BOUNDARY_VARIATION = 0.1; // ~19 mm
const double MAX_INNER_BOUNDARY_VARIATION = 0.0;

bool MDLHelper::computeThinPlateSplines(const std::string& sstrFileCollectionName, const std::string& sstrTextureCoordsFileName, const std::string& sstrOutFolder)
{
	std::cout << "outputThinPlateSplines start" << std::endl;

	//Load meshes
	FileLoader loader;

	std::vector<double> data;
	DataContainer mesh;
	std::vector<std::string> fileNames;
	std::vector<bool> missingShapes;
	size_t numExpressions(0);
	size_t numIdentities(0);
	if(!loader.loadIncompleteFileCollection(sstrFileCollectionName, data, mesh, fileNames, missingShapes, numExpressions, numIdentities))
	{
		std::cout << "Unable to load file collection " << sstrFileCollectionName << std::endl;
		return false;
	}

	const size_t dim = data.size()/(numExpressions*numIdentities);

	//Load texture coordinates
	std::vector<double> textureCoords;
	if(!loader.loadDataFile(sstrTextureCoordsFileName, textureCoords))
	{
		std::cout << "Unable to load texture coordinates " << sstrTextureCoordsFileName << std::endl;
		return false;
	}

	if(textureCoords.size() != 2*(dim/3))
	{
		std::cout << "Texture coordinates dimension does not fit " << textureCoords.size() << " != " << 2*(dim/3) << std::endl;
		return false;
	}

	const int numFiles = static_cast<int>(numIdentities*numExpressions);

#pragma omp parallel for
	for(int iFile = 0; iFile < numFiles; ++iFile)
	{
		std::cout << "File " << iFile+1 << " / " << numFiles << std::endl;

		if(missingShapes[iFile])
		{
			//Shape is missing
			continue;
		}

		const size_t startIndex = iFile*dim;

		std::vector<double> currData;
		currData.resize(dim, 0.0);
		
		for(size_t i = 0; i < dim; ++i)
		{
			currData[i] = data[startIndex+i];
		}

		std::vector<double> vecC; 
		std::vector<double> matA;
		std::vector<double> matW;
		if(!MathHelper::computeInterpolationBasis(textureCoords, 2, currData, 3, vecC, matA, matW))
		{
			std::cout << "Unable to compute thin-plate spline" << std::endl;
			continue;
		}

		const std::string sstrGeometryFileName = fileNames[iFile];
		const std::string sstrOutFileName = FileLoader::getFileName(sstrGeometryFileName) + ".tps";

		std::cout << "sstrGeometryFileName " << sstrGeometryFileName << std::endl;
		std::cout << "sstrOutFileName " << sstrOutFileName << std::endl;

		if(!FileWriter::saveThinPlateSpline(sstrOutFolder + "/" + sstrOutFileName, vecC, matA, matW, textureCoords))
		{
			std::cout << "Unable to save thin-plate spline " << sstrOutFileName << std::endl;
			continue;
		}
	}

	std::cout << "outputThinPlateSplines end" << std::endl;
	return true;
}

void MDLHelper::updateShapeParameter(const size_t iShape, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation)
{
	bool bSingleShapeParameters = (x.size() == 2*numVertices);

	const size_t paramOffset = iShape*2*numVertices;
	const size_t paramXOffset = bSingleShapeParameters ? 0 : paramOffset;

	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const size_t tmpParamStartIndex = 2*iVertex;

		double v1 = x[paramXOffset+tmpParamStartIndex];
		double v2 = x[paramXOffset+tmpParamStartIndex+1];
		const double tmpLength = sqrt(std::pow(v1,2) + std::pow(v2, 2));
		if(tmpLength > maxParamVariation)
		{
			const double factor = (maxParamVariation/tmpLength);
			v1 *= factor;
			v2 *= factor;
		}

		paramVariation[paramOffset+tmpParamStartIndex] = v1;
		paramVariation[paramOffset+tmpParamStartIndex+1] = v2;
	}
}

void MDLHelper::updateParameter(const size_t numSamples, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation)
{
	if(paramVariation.size() != 2*numSamples*numVertices)
	{
		std::cout << "Unable to update parameter " << paramVariation.size() << " != " << 2*numSamples*numVertices << std::endl;
	}

	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		updateShapeParameter(iSample, numVertices, maxParamVariation, x, paramVariation);
	}
}

void MDLHelper::updateShapeData(const size_t iShape, const std::vector<std::vector<double>>& vecCs, const std::vector<std::vector<double>>& matAs, const std::vector<std::vector<double>>& matWs, const std::vector<std::vector<double>>& sourcePointsVec
										, const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data)
{
	const size_t numSamples = vecCs.size();
	const size_t numVertices = data.size()/(3*numSamples);

	const size_t dataOffset = iShape*3*numVertices;
	const size_t paramOffset = iShape*2*numVertices;

	const std::vector<double>& currVecC = vecCs[iShape]; 
	const std::vector<double>& currMatA = matAs[iShape];
	const std::vector<double>& currMatW = matWs[iShape];
	const std::vector<double>& currSourcePoints = sourcePointsVec[iShape];

	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const size_t tmpDataStartIndex = 3*iVertex;
		const size_t tmpParamStartIndex = 2*iVertex;

		std::vector<double> paramVertex;
		paramVertex.resize(2, 0.0);

		paramVertex[0] = initialParametrization[paramOffset+tmpParamStartIndex]+paramVariation[paramOffset+tmpParamStartIndex];
		paramVertex[1] = initialParametrization[paramOffset+tmpParamStartIndex+1]+paramVariation[paramOffset+tmpParamStartIndex+1];

		std::vector<double> outDataVertex;
		if(!MathHelper::evaluateInterpolation(currVecC, currMatA, currMatW, currSourcePoints, paramVertex, outDataVertex))
		{
			std::cout << "MDLHelper::updateShapeData(...) - unable to compute tps point" << std::endl;
			return;
		}

		data[dataOffset+tmpDataStartIndex] = outDataVertex[0];
		data[dataOffset+tmpDataStartIndex+1] = outDataVertex[1];
		data[dataOffset+tmpDataStartIndex+2] = outDataVertex[2];
	}
}

void MDLHelper::updateData(std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
								  , const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data)
{
	const size_t numSamples = vecCs.size();
	const size_t numVertices = data.size()/(3*numSamples);
	
	if(data.size() != 3*numSamples*numVertices)
	{
		std::cout << "Unable to update data " << data.size() << " != " << 3*numSamples*numVertices << std::endl;
		return;
	}

	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		MDLHelper::updateShapeData(iSample, vecCs, matAs, matWs, sourcePointsVec, initialParametrization, paramVariation, data);
	}
}

void MDLHelper::computeShapeExcludedMean(const std::vector<double>& data, const size_t shapeDim, const size_t iExcludedShape, std::vector<double>& excludedShapeMean)
{
	excludedShapeMean.clear();
	excludedShapeMean.resize(shapeDim, 0.0);

	const size_t numSamples = data.size()/shapeDim;
	if(numSamples == 0)
	{
		return;
	}
	else if(numSamples == 1)
	{
		excludedShapeMean = data;
		return;
	}

	for(size_t i = 0; i < numSamples; ++i)
	{
		if(i == iExcludedShape)
		{
			continue;
		}

		const size_t startIndex = i*shapeDim;

		for(size_t j = 0; j < shapeDim; ++j)
		{
			const size_t currIndex = startIndex+j;
			excludedShapeMean[j] += data[currIndex];
		}
	}

	const double factor = 1.0 / static_cast<double>(numSamples-1);
	for(size_t i = 0; i < shapeDim; ++i)
	{
		excludedShapeMean[i] *= factor;
	}
}

bool MDLHelper::alignShapeData(const size_t iShape, const std::vector<double>& target, std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs)
{
	const size_t shapeDim = target.size();
	const size_t shapeOffset = iShape*shapeDim;

	std::vector<double> shapeData;
	shapeData.resize(shapeDim, 0.0);

	for(size_t j = 0; j < shapeDim; ++j)	
	{
		const size_t currIndex = shapeOffset+j;
		shapeData[j] = data[currIndex];
	}

	double s(0.0);
	std::vector<double> R; 
	std::vector<double> t;
	if(!MathHelper::computeAlignmentTrafo(shapeData, target, s, R, t, false))
	{
		std::cout << "Unable to compute alignment of shape " << iShape << std::endl;
		return false;
	}

	MathHelper::transformData(s, R, "N", t, "+", shapeData);
	MathHelper::transformThinPlateSpline(s, R, "N", t, "+", vecCs[iShape], matAs[iShape], matWs[iShape]);

	for(size_t j = 0; j < shapeDim; ++j)
	{
		const size_t currIndex = shapeOffset+j;
		data[currIndex] = shapeData[j];
	}

	return true;
}

bool MDLHelper::procrustesAlignShapeData(std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, const size_t numIter)
{
	const size_t numShapes = vecCs.size();
	const size_t dataDim = data.size()/numShapes;

	for(size_t iter = 0; iter < numIter; ++iter)
	{
		// Compute mean
		std::vector<double> procrustesMean;
		MathHelper::computeMean(data, dataDim, procrustesMean);

		// Compute alignment to the mean shape
		for(size_t iShape = 0; iShape < numShapes; ++iShape)
		{
			if(!MDLHelper::alignShapeData(iShape, procrustesMean, data, vecCs, matAs, matWs))
			{
				std::cout << "Unable to align shape data " << iShape << std::endl;
				return false;
			}
		}
	}

	return true;
}

void MDLHelper::outputParameterVariations(const size_t numSamples, const size_t numVertices, const std::vector<double>& parameterVariation, const std::string& sstrOutFolder, const std::string& sstrFileName)
{
	std::fstream outVariationStream(sstrOutFolder + "/" + sstrFileName, std::ios::out);

	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		const size_t paramOffset = iSample*2*numVertices;

		for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
		{
			const size_t tmpParamStartIndex = 2*iVertex;
			outVariationStream << parameterVariation[paramOffset+tmpParamStartIndex] << " " << parameterVariation[paramOffset+tmpParamStartIndex+1] << "   ";
		}

		outVariationStream << std::endl;
	}

	outVariationStream.close();
}

void MDLHelper::outputShapeData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames, const size_t iSample)
{
	const size_t numVertices = mesh.getNumVertices();
	const size_t numSamples = data.size()/(3*numVertices);
	if(fileNames.size() != numSamples)
	{
		std::cout << "Unable to output data " << fileNames.size() << " != " << numSamples << std::endl;
		return;
	}

	DataContainer currMesh = mesh;
	std::vector<double> vertices = currMesh.getVertexList();

	const size_t dataOffset = iSample*3*numVertices;

	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const size_t tmpDataStartIndex = 3*iVertex;
		vertices[tmpDataStartIndex] = data[dataOffset+tmpDataStartIndex];
		vertices[tmpDataStartIndex+1] = data[dataOffset+tmpDataStartIndex+1];
		vertices[tmpDataStartIndex+2] = data[dataOffset+tmpDataStartIndex+2];
	}

	currMesh.setVertexList(vertices);

	const std::string sstrCurrOutFileName = sstrOutFolder + "/" + fileNames[iSample];
	FileWriter::saveFile(sstrCurrOutFileName, currMesh);
}

void MDLHelper::outputData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames)
{
	const size_t numVertices = mesh.getNumVertices();
	const size_t numSamples = data.size()/(3*numVertices);
	if(fileNames.size() != numSamples)
	{
		std::cout << "Unable to output data " << fileNames.size() << " != " << numSamples << std::endl;
		return;
	}

	//TODO Run in parallel
	//for(size_t iSample = 0; iSample < numSamples; ++iSample)
#pragma omp parallel for
	for(int iSample = 0; iSample < numSamples; ++iSample)
	{
		outputShapeData(mesh, data, sstrOutFolder, fileNames, iSample);
	}

#ifdef OUTPUT_MEAN_FACE
	std::vector<double> mean;
	MathHelper::computeMean(data, 3*numVertices, mean);
	
	DataContainer currMesh = mesh;
	currMesh.setVertexList(mean);

	FileWriter::saveFile(sstrOutFolder + "/MeanFace.wrl", currMesh);
#endif

}

void MDLHelper::precomputeBiLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights)
{
	//Compute for each vertex all its neighbors
	const size_t numVertices = mesh.getNumVertices();
	precomputedSmoothnessIndices.resize(numVertices);
	precomputedSmoothnessWeights.resize(numVertices);

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	const size_t numTriangles = vertexIndexList.size();

	std::vector<std::set<int>> vertexNeighbors;
	vertexNeighbors.resize(numVertices);

	for(size_t i = 0; i < numTriangles; ++i)
	{
		const std::vector<int>& currTriangleIndices = vertexIndexList[i];
		const int i1 = currTriangleIndices[0];
		const int i2 = currTriangleIndices[1];
		const int i3 = currTriangleIndices[2];

		vertexNeighbors[i1].insert(i2);
		vertexNeighbors[i1].insert(i3);

		vertexNeighbors[i2].insert(i1);
		vertexNeighbors[i2].insert(i3);

		vertexNeighbors[i3].insert(i1);
		vertexNeighbors[i3].insert(i2);
	}

	//Remove points that are in the 2-ring neighborhood of the boundary
	std::vector<bool> invalidSmoothNeighbors;
	invalidSmoothNeighbors.resize(numVertices, false);

	std::vector<size_t> boundaryVertices;
	MathHelper::computeBoundaryVertices(mesh, boundaryVertices);

	const size_t numBoundaryVertices = boundaryVertices.size();
	for(size_t i = 0; i < numBoundaryVertices; ++i)
	{
		const size_t currVertex = boundaryVertices[i];
		invalidSmoothNeighbors[currVertex] = true;

		const std::set<int>& borderNeighbors = vertexNeighbors[currVertex];
		std::set<int>::const_iterator currNeighborIter = borderNeighbors.begin();
		const std::set<int>::const_iterator endNeighborIter = borderNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			//Boundary points in the 1-ring neighborhood.
			const size_t borderNeighborID = *currNeighborIter;
			invalidSmoothNeighbors[borderNeighborID] = true;

			//const std::set<int>& border2RingNeighbors = vertexNeighbors[borderNeighborID];
			//std::set<int>::const_iterator curr2RingNeighborIter = border2RingNeighbors.begin();
			//const std::set<int>::const_iterator end2RingNeighborIter = border2RingNeighbors.end();
			//for(; curr2RingNeighborIter != end2RingNeighborIter; ++curr2RingNeighborIter)
			//{
			//	//Boundary points in the 2-ring neighborhood.
			//	const size_t border2RingNeighborID = *curr2RingNeighborIter;
			//	invalidSmoothNeighbors[border2RingNeighborID] = true;
			//}
		}
	}

	//Compute weights for all vertices
	for(int iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		if(invalidSmoothNeighbors[iVertex])
		{
			continue;
		}

		const std::set<int>& currNeighbors = vertexNeighbors[iVertex];
		if(currNeighbors.empty())
		{
			continue;
		}

		std::map<int, double> indexWeightMap; 
		indexWeightMap.insert(std::make_pair(iVertex, 1.0));
		
		std::map<int, double>::iterator mapIter;

		const double neighborFactor = -2.0/static_cast<double>(currNeighbors.size());

		//Iterate over all neighbors
		std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
		std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			const int currNeighborIndex = *currNeighborIter;

			mapIter = indexWeightMap.find(currNeighborIndex);
			if(mapIter != indexWeightMap.end())
			{
				mapIter->second += neighborFactor;
			}
			else
			{
				indexWeightMap.insert(std::make_pair(currNeighborIndex, neighborFactor));
			}

			//Iterate over all neighbor neighbors
			const std::set<int>& neighborNeighbors = vertexNeighbors[currNeighborIndex];
			if(neighborNeighbors.empty())
			{
				continue;
			}

			const double neighborNeighborFactor = 1.0/static_cast<double>(currNeighbors.size()*neighborNeighbors.size());

			std::set<int>::const_iterator neighborNeighborIter = neighborNeighbors.begin();
			std::set<int>::const_iterator endNeighborNeighborIter = neighborNeighbors.end();
			for(; neighborNeighborIter != endNeighborNeighborIter; ++neighborNeighborIter)
			{
				const int neighborNeighborIndex = *neighborNeighborIter;

				mapIter = indexWeightMap.find(neighborNeighborIndex);
				if(mapIter != indexWeightMap.end())
				{
					mapIter->second += neighborNeighborFactor;
				}
				else
				{
					indexWeightMap.insert(std::make_pair(neighborNeighborIndex, neighborNeighborFactor));
				}
			}
		}

		const size_t numElements = indexWeightMap.size();
		std::vector<size_t>& currSmoothnessIndices = precomputedSmoothnessIndices[iVertex];
		currSmoothnessIndices.reserve(numElements);

		std::vector<double>& currSmoothnessWeights = precomputedSmoothnessWeights[iVertex];
		currSmoothnessWeights.reserve(numElements);

		for(mapIter=indexWeightMap.begin(); mapIter != indexWeightMap.end(); ++mapIter)
		{
			currSmoothnessIndices.push_back(mapIter->first);
			currSmoothnessWeights.push_back(mapIter->second);
		}
	}
}

void MDLHelper::precomputeLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights)
{
	//Compute for each vertex all its neighbors
	const size_t numVertices = mesh.getNumVertices();
	precomputedSmoothnessIndices.resize(numVertices);
	precomputedSmoothnessWeights.resize(numVertices);

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	const size_t numTriangles = vertexIndexList.size();

	std::vector<std::set<int>> vertexNeighbors;
	vertexNeighbors.resize(numVertices);

	for(size_t i = 0; i < numTriangles; ++i)
	{
		const std::vector<int>& currTriangleIndices = vertexIndexList[i];
		const int i1 = currTriangleIndices[0];
		const int i2 = currTriangleIndices[1];
		const int i3 = currTriangleIndices[2];

		vertexNeighbors[i1].insert(i2);
		vertexNeighbors[i1].insert(i3);

		vertexNeighbors[i2].insert(i1);
		vertexNeighbors[i2].insert(i3);

		vertexNeighbors[i3].insert(i1);
		vertexNeighbors[i3].insert(i2);
	}

	//Remove points that are in the 2-ring neighborhood of the boundary
	std::vector<bool> invalidSmoothNeighbors;
	invalidSmoothNeighbors.resize(numVertices, false);

	std::vector<size_t> boundaryVertices;
	MathHelper::computeBoundaryVertices(mesh, boundaryVertices);

	const size_t numBoundaryVertices = boundaryVertices.size();
	for(size_t i = 0; i < numBoundaryVertices; ++i)
	{
		const size_t currVertex = boundaryVertices[i];
		invalidSmoothNeighbors[currVertex] = true;

		//const std::set<int>& borderNeighbors = vertexNeighbors[currVertex];
		//std::set<int>::const_iterator currNeighborIter = borderNeighbors.begin();
		//const std::set<int>::const_iterator endNeighborIter = borderNeighbors.end();
		//for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		//{
		//	//Boundary points in the 1-ring neighborhood.
		//	const size_t borderNeighborID = *currNeighborIter;
		//	invalidSmoothNeighbors[borderNeighborID] = true;
		//}
	}

	//Compute weights for all vertices
	for(int iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		if(invalidSmoothNeighbors[iVertex])
		{
			continue;
		}

		const std::set<int>& currNeighbors = vertexNeighbors[iVertex];
		if(currNeighbors.empty())
		{
			continue;
		}

		std::map<int, double> indexWeightMap; 
		indexWeightMap.insert(std::make_pair(iVertex, -1.0));
		
		std::map<int, double>::iterator mapIter;

		const double neighborFactor = 1.0/static_cast<double>(currNeighbors.size());

		//Iterate over all neighbors
		std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
		std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			const int currNeighborIndex = *currNeighborIter;

			mapIter = indexWeightMap.find(currNeighborIndex);
			if(mapIter != indexWeightMap.end())
			{
				mapIter->second += neighborFactor;
			}
			else
			{
				indexWeightMap.insert(std::make_pair(currNeighborIndex, neighborFactor));
			}

			//Iterate over all neighbor neighbors
			const std::set<int>& neighborNeighbors = vertexNeighbors[currNeighborIndex];
			if(neighborNeighbors.empty())
			{
				continue;
			}
		}

		const size_t numElements = indexWeightMap.size();
		std::vector<size_t>& currSmoothnessIndices = precomputedSmoothnessIndices[iVertex];
		currSmoothnessIndices.reserve(numElements);

		std::vector<double>& currSmoothnessWeights = precomputedSmoothnessWeights[iVertex];
		currSmoothnessWeights.reserve(numElements);

		for(mapIter=indexWeightMap.begin(); mapIter != indexWeightMap.end(); ++mapIter)
		{
			currSmoothnessIndices.push_back(mapIter->first);
			currSmoothnessWeights.push_back(mapIter->second);
		}
	}
}

void MDLHelper::precomputeVertexLaplacians(const std::vector<double>& data, const size_t numShapes, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices
														, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, std::vector<std::vector<double>>& vertexLaplacians)
{
	const size_t numVertices = precomputedSmoothnessIndices.size();
	if(data.size() != 3*numVertices*numShapes)
	{
		return;
	}

	vertexLaplacians.resize(numShapes);

	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		//Vertex laplacians are only required for missing and corrupt shapes
		if(!missingShapes[iShape] && !corruptShapes[iShape])
			continue;

		MDLHelper::precomputeVertexLaplacian(data, iShape, precomputedSmoothnessIndices, precomputedSmoothnessWeights, vertexLaplacians[iShape]);
	}
}

void MDLHelper::precomputeVertexLaplacian(const std::vector<double>& data, const size_t iShape, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights
														, std::vector<double>& vertexLaplacian)
{
	const size_t numVertices = precomputedSmoothnessIndices.size();

	vertexLaplacian.clear();
	vertexLaplacian.resize(3*numVertices, 0.0);

	const size_t shapeOffset = iShape*3*numVertices;

	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const std::vector<size_t>& currSmoothnessIndices = precomputedSmoothnessIndices[iVertex];
		const std::vector<double>& currSmoothnessWeights = precomputedSmoothnessWeights[iVertex];

		if(currSmoothnessIndices.empty())
		{
			continue;
		}

		const size_t currNumSmoothnessIndices = currSmoothnessIndices.size();
		for(size_t j = 0; j < currNumSmoothnessIndices; ++j)
		{
			const size_t currVertexStartIndex = 3*currSmoothnessIndices[j];
			const size_t currSmoothStartIndex = shapeOffset + currVertexStartIndex;
			const double currSmoothWeight = currSmoothnessWeights[j];

			vertexLaplacian[3*iVertex] += currSmoothWeight*data[currSmoothStartIndex];
			vertexLaplacian[3*iVertex+1] += currSmoothWeight*data[currSmoothStartIndex+1];
			vertexLaplacian[3*iVertex+2] += currSmoothWeight*data[currSmoothStartIndex+2];
		}
	}
}

void MDLHelper::robustModelLearning(const std::string& sstrFileCollectionName, const std::string& sstrLmkIndexFileName, const std::string & sstCorruptFileCollectionName 
											, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName, const std::string& sstrOutFolder)
{
	//1. Load data
	std::vector<double> data;
	std::vector<std::string> geometryFileNames; 
	size_t numIdentities(0);
	size_t numExpressions(0);
	DataContainer mesh;
	std::vector<size_t> modelLmkIndices;
	
	std::vector<bool> missingShapes;
	std::vector<bool> corruptShapes; 
	std::vector<std::vector<double>> corruptData;
	std::vector<std::vector<std::pair<double,bool>>> curruptDataLmks;
	std::vector<std::vector<double>> corruptDataNormals;
	
	std::vector<std::vector<double>> vecCs;
	std::vector<std::vector<double>> matAs;
	std::vector<std::vector<double>> matWs;
	std::vector<std::vector<double>> sourcePointsVec; 
	
	std::vector<size_t> outerBoundaryVertexIDs;
	std::vector<size_t> innerBoundaryVertexIDs;

	if(!MDLHelper::getRMMData(sstrFileCollectionName, sstrLmkIndexFileName, sstrOuterBoundaryIndexFileName, sstrInnerBoundaryIndexFileName, sstCorruptFileCollectionName,  data, geometryFileNames, numIdentities, numExpressions, mesh, modelLmkIndices
									, missingShapes, corruptShapes, corruptData, curruptDataLmks, corruptDataNormals, vecCs, matAs, matWs, sourcePointsVec, outerBoundaryVertexIDs, innerBoundaryVertexIDs))
	{
		std::cout << "Unable to load RMM data" << std::endl;
		return;
	}

	const size_t numShapes = numExpressions*numIdentities;
	const size_t numShapeVertices = data.size()/(3*numShapes);
	const size_t shapeDataDim = 3*numShapeVertices;
	const size_t shapeParameterDim = 2*numShapeVertices;

	std::vector<size_t> modeDim;
	modeDim.push_back(shapeDataDim);
	modeDim.push_back(numIdentities);
	modeDim.push_back(numExpressions);

	//2. Generate current output folder and output config file
	const std::string sstrCurrOutFolder = MDLHelper::getConfigOutFolder(sstrOutFolder);
	FileWriter::makeDirectory(sstrCurrOutFolder);
	MDLHelper::outputConfig(sstrCurrOutFolder); //TODO: Update config file for RMM

	//3. Output missing and corrupt data file info
	std::fstream infoOut(sstrCurrOutFolder + "/DataInfo.txt", std::ios::out);
	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		if(missingShapes[iShape])
			infoOut << geometryFileNames[iShape] << " missing" << std::endl;

		if(corruptShapes[iShape])
			infoOut << geometryFileNames[iShape] << " corrupt" << std::endl;
	}
	infoOut.close();

	//4. Initialize semantic correspondence
	std::vector<size_t> semCorr;
	semCorr.resize(numShapes);
	for(size_t i = 0; i < numShapes; ++i)
	{
		semCorr[i] = i;
	}

	//5. Initialize missing and corrupt shapes
	MDLHelper::initializeMissingData(data, modeDim, missingShapes, corruptShapes);

#ifdef OUTPUT_INITIAL_ALIGNED_DATA
	const std::string sstrInitialOutFolder = sstrCurrOutFolder + "/Initial";
	FileWriter::makeDirectory(sstrInitialOutFolder);
	
	MDLHelper::outputData(mesh, data, sstrInitialOutFolder, geometryFileNames);
	MDLHelper::outputCompactness(data, semCorr, shapeDataDim, numIdentities, numExpressions, sstrInitialOutFolder + "/CompactnessInit.txt");
#endif //OUTPUT_INITIAL_ALIGNED_DATA

	//6. Initialize corrupt data kd-trees 
	std::vector<KDTree3*> corrShapesKDTrees;
	MDLHelper::computeKDTrees(numShapes, corruptShapes, corruptData, corrShapesKDTrees);
	
	//7. Initialize corrupt data rigid alignments (computed using the provided landmarks)
	std::vector<double> corrShapes_s;
	std::vector<std::vector<double>> corrShapes_R;
	std::vector<std::vector<double>> corrShapes_t;
	if(!MDLHelper::computeCorruptDataLmkAlignment(data, numShapes, shapeDataDim, corruptShapes, modelLmkIndices, curruptDataLmks, corrShapes_s, corrShapes_R, corrShapes_t))
	{
		std::cout << "Unable to compute rigid landmark alignment for corrupt data" << std::endl;
		return;
	}

	//8. Initialize parameter
	std::vector<double> initialParam;
	MDLHelper::convertParameters(sourcePointsVec, numShapes, shapeParameterDim, missingShapes, corruptShapes, initialParam);

	//9. Initialize parameter variation used for correspondence optimization
	std::vector<double> parameterVariation;
	parameterVariation.resize(numShapes*shapeParameterDim, 0.0);

	//10. Pre-compute smoothness values
	std::vector<std::vector<size_t>> precomputedSmoothnessIndices;
	std::vector<std::vector<double>> precomputedSmoothnessWeights;

#ifdef USE_BI_LAPLACIAN_SMOOTHNESS
	precomputeBiLaplacianSmoothnessWeights(mesh, precomputedSmoothnessIndices, precomputedSmoothnessWeights);
#else
	precomputeLaplacianSmoothnessWeights(mesh, precomputedSmoothnessIndices, precomputedSmoothnessWeights);
#endif //USE_BI_LAPLACIAN_SMOOTHNESS

	std::vector<std::vector<double>> vertexLaplacians;
	MDLHelper::precomputeVertexLaplacians(data, numShapes, missingShapes, corruptShapes, precomputedSmoothnessIndices, precomputedSmoothnessWeights, vertexLaplacians);

	//11. Compute boundaries for bounded Quasi-Newton
	vnl_vector<long> completionBoundSelection;
	vnl_vector<long> optimizationBoundSelection;
	std::vector<vnl_vector<double>> lowerShapeBounds;
	std::vector<vnl_vector<double>> upperShapeBounds;
	if(!computeRMMOptimizationBounds(numShapes, shapeDataDim, shapeParameterDim, missingShapes, corruptShapes, initialParam, outerBoundaryVertexIDs, innerBoundaryVertexIDs, completionBoundSelection, optimizationBoundSelection, lowerShapeBounds, upperShapeBounds))
	{
		std::cout << "Unable to compute optimization bounds" << std::endl;
		return;
	}

#ifdef OUTPUT_TIME
	const double startTime = PerformanceCounter::getTime();

	std::fstream outTime(sstrCurrOutFolder + "/Time.txt", std::ios::out);
	outTime.close();
#endif

	//Start the optimization
	for(size_t iter = 0; iter < NUM_ITERATION; ++iter)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "*******************************" << std::endl;
		std::cout << "*******************************" << std::endl;
		std::cout << "Iteration " << iter+1 << std::endl;
		std::cout << "*******************************" << std::endl;
		std::cout << "*******************************" << std::endl;
#endif //DEBUG_OUTPUT

#ifdef OUTPUT_TIME
		const double iterStartTime = PerformanceCounter::getTime();
#endif

#ifdef OPTIMIZE_SEMANTIC_CORRESPONDENCE
		//Optimize semantic correspondence
		std::vector<size_t> tmpSemCorr;
		if(!RelabelMultilinearData::optimizeSemanticCorrespondence(data, modeDim, NUM_SEM_CORR_OPT, semCorr, tmpSemCorr))
		{
			std::cout << "Optimizing semantic correspondence failed" << std::endl;
		}
		else
		{
			semCorr = tmpSemCorr;
		}
#endif //OPTIMIZE_SEMANTIC_CORRESPONDENCE

#ifdef OUTPUT_TIME
		const double semCorrStartTime = PerformanceCounter::getTime();
#endif //OUTPUT_TIME

#ifdef OUTPUT_ITERATION_RESULTS
		const std::string sstrIterOutFolder = sstrCurrOutFolder + "/Iter_" + MathHelper::convertToStdString(iter+1);
		FileWriter::makeDirectory(sstrIterOutFolder);

#ifdef OUTPUT_TIME
		const double semCorrEndTime = PerformanceCounter::getTime();
		const double semCorrTime = semCorrEndTime - semCorrStartTime;
		
		std::cout << std::endl;
		std::cout << "Semantic correspondence optimization time: " << semCorrTime <<  "s  " << semCorrTime/60.0 <<  "min" << std::endl;
		std::cout << std::endl;

		std::fstream outShapeTime(sstrCurrOutFolder + "/ShapeTime.txt", std::ios::app);
		outShapeTime << "Semantic correspondence optimization time: " << semCorrTime <<  "s  " << semCorrTime/60.0 <<  "min" << std::endl;;
		outShapeTime.close();
#endif //OUTPUT_TIME

		RelabelMultilinearData::outputPermutationMatrix(semCorr, numIdentities, numExpressions, sstrIterOutFolder + "/PermutationMatrixIter.txt");

		if(!FileWriter::writeFileCollection(sstrIterOutFolder + "/FileCollection.txt", geometryFileNames, semCorr, numIdentities, numExpressions))
		{
			std::cout << "Unable to write file collection" << std::endl;
		}
#endif //OUTPUT_ITERATION_RESULTS

		//Shape-wise optimization
		std::vector<int> permutedShapeIndices;
		MathHelper::getRandomlyPermutedInteger(0, numShapes-1, permutedShapeIndices);
		
		for(size_t i = 0; i < numShapes; ++i)
		{
			const size_t iShape = permutedShapeIndices[i];
			const size_t shapeIndex = semCorr[iShape];

#ifdef DEBUG_OUTPUT
			std::cout << std::endl;
			std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
			std::cout << "Optimization of shape " << shapeIndex << " (" << i+1 << " of " << numShapes << ")" << std::endl;
			std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
			std::cout << std::endl;
#endif //DEBUG_OUTPUT

#ifdef OUTPUT_TIME
			const double shapeStartTime = PerformanceCounter::getTime();
#endif //OUTPUT_TIME

			if(missingShapes[shapeIndex])
			{
				//Estimate missing shape
				MDLHelper::optimizeMissingShape(data, modeDim, semCorr, missingShapes, corruptShapes, precomputedSmoothnessIndices, precomputedSmoothnessWeights, vertexLaplacians[shapeIndex], completionBoundSelection, lowerShapeBounds[shapeIndex], upperShapeBounds[shapeIndex], iShape);
			}
			else if(corruptShapes[shapeIndex])
			{
				//Reconstruct corrupt shape

				//Refine rigid alignment by a few rigid ICP steps
				std::vector<double> currShape;
				currShape.resize(shapeDataDim, 0.0);

				const size_t shapeOffset = shapeIndex*shapeDataDim;
				for(size_t j = 0; j < shapeDataDim; ++j)
				{
					currShape[j] = data[shapeOffset+j];
				}

				double& s = corrShapes_s[shapeIndex];
				std::vector<double>& R = corrShapes_R[shapeIndex];
				std::vector<double>& t = corrShapes_t[shapeIndex];
				MathHelper::transformData(s, R, "N", t, "+", currShape);

				double tmp_s(1.0);
				std::vector<double> tmp_R;
				std::vector<double> tmp_t;
				if(MathHelper::computeICPAlignment(currShape, corruptData[shapeIndex], corruptDataNormals[shapeIndex], *(corrShapesKDTrees[shapeIndex]), MAX_CORR_DATA_DIST, tmp_s, tmp_R, tmp_t, false))
				{
					std::cout << "Alignment successfull" << std::endl;

					// s = s_new * s_prev
					s *= tmp_s;

					// R = R_new * R_prev
					MathHelper::rotateData(tmp_R, "N", R);
		
					// t = s_new * R_new * t_prev + t_new
					MathHelper::rotateData(tmp_R, "N", t);
					MathHelper::scaleData(tmp_s, t);
					MathHelper::translateData(tmp_t,  "+", t);					
				}
				else
				{
					std::cout << "Failed refining alignment" << std::endl;
				}

				MDLHelper::optimizeCorruptShape(data, modeDim, semCorr, missingShapes, corruptShapes, corruptData[shapeIndex], corruptDataNormals[shapeIndex], corrShapesKDTrees[shapeIndex], MAX_CORR_DATA_DIST, s, R, t
														, precomputedSmoothnessIndices, precomputedSmoothnessWeights, vertexLaplacians[shapeIndex], completionBoundSelection, lowerShapeBounds[shapeIndex], upperShapeBounds[shapeIndex], iShape);
			}
			else
			{
#ifdef OPTIMIZE_VERTEX_CORRESPONDENCE
				//Optimize vertex correspondence
				MDLHelper::optimizeCorrespondence(data, initialParam, parameterVariation, modeDim, semCorr, vecCs, matAs, matWs, sourcePointsVec, precomputedSmoothnessIndices, precomputedSmoothnessWeights, optimizationBoundSelection
															, lowerShapeBounds[shapeIndex], upperShapeBounds[shapeIndex], iShape);
#else
				continue;
#endif //OPTIMIZE_VERTEX_CORRESPONDENCE
			}

#ifdef OUTPUT_TIME
			const double shapeEndTime = PerformanceCounter::getTime();

			const double shapeTime = shapeEndTime-shapeStartTime;
			std::cout << std::endl;
			std::cout << "Shape optimization time: " << shapeTime <<  "s  " << shapeTime/60.0 <<  "min ";

			std::fstream outShapeTime(sstrCurrOutFolder + "/ShapeTime.txt", std::ios::app);
			outShapeTime << "Shape " <<  iShape << " (" << i+1 << " of " << numShapes << ") - Optimization time: " << shapeTime <<  "s  " << shapeTime/60.0 <<  "min ";
			if(missingShapes[shapeIndex])
			{
				std::cout << "(missing shape)" << std::endl;
				outShapeTime << "(missing shape)" << std::endl;
			}
			else if(corruptShapes[shapeIndex])
			{
				std::cout << "(corrupt shape)" << std::endl;
				outShapeTime << "(corrupt shape)" << std::endl;
			}
			else
			{
				std::cout << "(correspondence opt.)" << std::endl;
				outShapeTime << "(correspondence opt.)" << std::endl;
			}
			outShapeTime.close();
#endif //OUTPUT_TIME

#ifdef OUTPUT_ITERATION_RESULTS
			//Output the currently optimized shape
			outputShapeData(mesh, data, sstrIterOutFolder, geometryFileNames, shapeIndex);
#endif //OUTPUT_ITERATION_RESULTS
		}

#ifdef OUTPUT_TIME
		const double iterEndTime = PerformanceCounter::getTime();
		const double iterTime = iterEndTime-iterStartTime; 

		std::fstream outIterTime(sstrIterOutFolder + "/TimeIter.txt", std::ios::out);
		outIterTime << "Iteration time: " << iterTime <<  "s  " << iterTime/60.0 <<  "min  " << iterTime/3600.0 <<  "h" << std::endl;
		outIterTime.close();

		outTime.open(sstrCurrOutFolder + "/Time.txt", std::ios::app);
		outTime << "Iteration " << iter+1 << std::endl;
		outTime << "Iteration time: " << iterTime <<  "s  " << iterTime/60.0 <<  "min  " << iterTime/3600.0 <<  "h" << std::endl;
		outTime << std::endl;
		outTime.close();
#endif //OUTPUT_TIME

#ifdef OUTPUT_ITERATION_RESULTS
		//Output the parameter variations for all files (for missing and corrupt shapes the variations are just zeros)
		MDLHelper::outputParameterVariations(numShapes, numShapeVertices, parameterVariation, sstrIterOutFolder, "VariationIter.txt");

		//Output the file collection for the current semantic correspondence
		if(!FileWriter::writeFileCollection(sstrIterOutFolder + "/FileCollectionIter.txt", geometryFileNames, semCorr, numIdentities, numExpressions))
		{
			std::cout << "Unable to write file collection" << std::endl;
		}

		MDLHelper::outputCompactness(data, semCorr, shapeDataDim, numIdentities, numExpressions, sstrIterOutFolder + "/CompactnessIter.txt");
#endif //OUTPUT_ITERATION_RESULTS
	}

#ifdef OUTPUT_TIME
	const double endTime = PerformanceCounter::getTime();
	const double time = endTime-startTime; 

	outTime.open(sstrCurrOutFolder + "/Time.txt", std::ios::app);
	outTime << "Overall time " << time <<  "s  " << time/60.0 <<  "min  " << time/3600.0 <<  "h" << std::endl;
	outTime.close();
#endif //OUTPUT_TIME

#ifdef OUTPUT_COMPACTNESS
	MDLHelper::outputCompactness(data, semCorr, shapeDataDim, numIdentities, numExpressions, sstrCurrOutFolder + "/CompactnessEnd.txt");
#endif //OUTPUT_COMPACTNESS

	MDLHelper::outputParameterVariations(numShapes, numShapeVertices, parameterVariation, sstrCurrOutFolder, "VariationX.txt");
	MDLHelper::outputData(mesh, data, sstrCurrOutFolder, geometryFileNames);

	if(!FileWriter::writeFileCollection(sstrCurrOutFolder + "/FileCollection.txt", geometryFileNames, semCorr, numIdentities, numExpressions))
	{
		std::cout << "Unable to write file collection" << std::endl;
	}

	//Cleanup
	for(size_t i = 0; i < numShapes; ++i)
	{
		if(corrShapesKDTrees[i] != NULL)
		{
			delete corrShapesKDTrees[i];
			corrShapesKDTrees[i] = NULL;
		}
	}
}

void MDLHelper::optimizeMissingShape(std::vector<double>& data, const std::vector<size_t>& modeDim, const std::vector<size_t>& semCorr, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes
											, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const std::vector<double>& vertexLaplacians, const vnl_vector<long>& boundSelection
											, const vnl_vector<double>& lowerShapeBounds, const vnl_vector<double>& upperShapeBounds, const size_t iShape)
{
	const size_t shapeIndex = semCorr[iShape];
	const size_t shapeDataDim = modeDim[0];

	TensorCompletionCostFunction fkt(data, modeDim[1], modeDim[2], semCorr, missingShapes, corruptShapes, IDENTITY_WEIGHT, EXPRESSION_WEIGHT, iShape);
	fkt.setSmoothnessValues(precomputedSmoothnessIndices, precomputedSmoothnessWeights, DATA_COMPLETION_SMOOTHNESS_WEIGHT, vertexLaplacians);

	vnl_lbfgsb minimizer(fkt);
	minimizer.set_cost_function_convergence_factor(1000); 
	minimizer.set_projected_gradient_tolerance(0.000000000001);		

	minimizer.set_max_function_evals(NUM_FKT_EVAL);
	
	minimizer.set_bound_selection(boundSelection);
	minimizer.set_lower_bound(lowerShapeBounds);
	minimizer.set_upper_bound(upperShapeBounds);

#ifdef DEBUG_OUTPUT
	minimizer.set_trace(true);
#endif //DEBUG_OUTPUT

	vnl_vector<double> x(shapeDataDim, 0.0);
	minimizer.minimize(x);

	if(minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_FTOL
		|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XTOL
		|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XFTOL
		|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_GTOL)
	{
		const size_t shapeOffset = shapeIndex*shapeDataDim;
		for(size_t i = 0; i < shapeDataDim; ++i)
		{
			data[shapeOffset+i] += x[i];
		}
	}
	else if(minimizer.get_failure_code() == vnl_lbfgsb::FAILED_TOO_MANY_ITERATIONS)
	{
		std::cout << "Reached maximum number of function evaluations " << minimizer.get_failure_code() << std::endl;
		if(minimizer.obj_value_reduced())
		{
			std::cout << "Function value reduced" << std::endl;

			const size_t shapeOffset = shapeIndex*shapeDataDim;
			for(size_t i = 0; i < shapeDataDim; ++i)
			{
				data[shapeOffset+i] += x[i];
			}
		}
		else
		{
			std::cout << "****************************************************" << std::endl;
			std::cout << "Function value not reduced" << std::endl;
			std::cout << "****************************************************" << std::endl;
		}
	}
	else
	{
		std::cout << "****************************************************" << std::endl;
		std::cout << "Minimizer failed convergence " << minimizer.get_failure_code() << std::endl;
		std::cout << "****************************************************" << std::endl;
	}
}

void MDLHelper::optimizeCorruptShape(std::vector<double>& data, const std::vector<size_t>& modeDim, const std::vector<size_t>& semCorr, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const std::vector<double>& corruptData
											, const std::vector<double>& corruptDataNormals, const KDTree3* pCorrShapesKDTree, const double maxCorrDataDist, const double s, const std::vector<double>& R, const std::vector<double>& t
											, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const std::vector<double>& vertexLaplacians, const vnl_vector<long>& boundSelection
											, const vnl_vector<double>& lowerShapeBounds, const vnl_vector<double>& upperShapeBounds, const size_t iShape)
{
	const size_t shapeIndex = semCorr[iShape];
	const size_t shapeDataDim = modeDim[0];

	TensorCompletionCostFunction fkt(data, modeDim[1], modeDim[2], semCorr, missingShapes, corruptShapes, corruptData, corruptDataNormals, *pCorrShapesKDTree, maxCorrDataDist, s, R, t, IDENTITY_WEIGHT, EXPRESSION_WEIGHT, CORRUPT_DATA_WEIGHT, iShape);
	fkt.setSmoothnessValues(precomputedSmoothnessIndices, precomputedSmoothnessWeights, DATA_COMPLETION_SMOOTHNESS_WEIGHT, vertexLaplacians);

	vnl_lbfgsb minimizer(fkt);
	minimizer.set_cost_function_convergence_factor(1000); 
	minimizer.set_projected_gradient_tolerance(0.000000000001);		

	minimizer.set_max_function_evals(NUM_FKT_EVAL);
	
	minimizer.set_bound_selection(boundSelection);
	minimizer.set_lower_bound(lowerShapeBounds);
	minimizer.set_upper_bound(upperShapeBounds);

#ifdef DEBUG_OUTPUT
	minimizer.set_trace(true);
#endif //DEBUG_OUTPUT

	vnl_vector<double> x(shapeDataDim, 0.0);
	minimizer.minimize(x);

	if(minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_FTOL
		|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XTOL
		|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XFTOL
		|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_GTOL)
	{
		const size_t shapeOffset = shapeIndex*shapeDataDim;
		for(size_t i = 0; i < shapeDataDim; ++i)
		{
			data[shapeOffset+i] += x[i];
		}
	}
	else if(minimizer.get_failure_code() == vnl_lbfgsb::FAILED_TOO_MANY_ITERATIONS)
	{
		std::cout << "Reached maximum number of function evaluations " << minimizer.get_failure_code() << std::endl;
		if(minimizer.obj_value_reduced())
		{
			std::cout << "Function value reduced" << std::endl;

			const size_t shapeOffset = shapeIndex*shapeDataDim;
			for(size_t i = 0; i < shapeDataDim; ++i)
			{
				data[shapeOffset+i] += x[i];
			}
		}
		else
		{
			std::cout << "****************************************************" << std::endl;
			std::cout << "Function value not reduced" << std::endl;
			std::cout << "****************************************************" << std::endl;
		}
	}
	else
	{
		std::cout << "****************************************************" << std::endl;
		std::cout << "Minimizer failed convergence " << minimizer.get_failure_code() << std::endl;
		std::cout << "****************************************************" << std::endl;
	}
}

void MDLHelper::optimizeCorrespondence(std::vector<double>& data, const std::vector<double>& initialParam, std::vector<double>& parameterVariation, const std::vector<size_t>& modeDim, const std::vector<size_t>& semCorr
													, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
													, const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights
													, const vnl_vector<long>& boundSelection, const vnl_vector<double>& lowerShapeBounds, const vnl_vector<double>& upperShapeBounds, const size_t iShape)
{
	const size_t shapeIndex = semCorr[iShape];
	const size_t numShapeVertices = modeDim[0]/3;
	const size_t shapeDataDim = modeDim[0];
	const size_t shapeParameterDim = 2*numShapeVertices;

	//Compute mean for all faces except current shape
	std::vector<double> excludedShapeMean;
	computeShapeExcludedMean(data, shapeDataDim, shapeIndex, excludedShapeMean);

	const size_t numSubIter = NUM_FKT_EVAL/NUM_NUM_FKT_EVAL_ALIGNMENT;
	for(size_t subIter = 0; subIter < numSubIter; ++subIter)
	{
		MDLShapeCostFunction fkt(data, initialParam, modeDim[1], modeDim[2], vecCs, matAs, matWs, sourcePointsVec, semCorr, IDENTITY_WEIGHT, EXPRESSION_WEIGHT, MAX_PARAMETER_VARIATION, iShape);
		fkt.setSmoothnessValues(precomputedSmoothnessIndices, precomputedSmoothnessWeights, CORRESPONDENCE_OPTIMIZATION_SMOOTHNESS_WEIGHT);

		vnl_lbfgsb minimizer(fkt);
		minimizer.set_cost_function_convergence_factor(10000000); 
		minimizer.set_projected_gradient_tolerance(0.00001);		
		minimizer.set_max_function_evals(NUM_FKT_EVAL/numSubIter);

		minimizer.set_bound_selection(boundSelection);
		minimizer.set_lower_bound(lowerShapeBounds);
		minimizer.set_upper_bound(upperShapeBounds);

#ifdef DEBUG_OUTPUT
		minimizer.set_trace(true);
#endif //DEBUG_OUTPUT

		const size_t paramOffset = shapeIndex*shapeParameterDim;

		vnl_vector<double> x(shapeParameterDim, 0.0);
		for(size_t j = 0; j < shapeParameterDim; ++j)
		{
			x[j] = parameterVariation[paramOffset+j];
		}		

		minimizer.minimize(x);

		if(minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_FTOL
			|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XTOL
			|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XFTOL
			|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_GTOL)
		{
			MDLHelper::updateShapeParameter(shapeIndex, numShapeVertices, MAX_PARAMETER_VARIATION, x, parameterVariation);
		}
		else if(minimizer.get_failure_code() == vnl_lbfgsb::FAILED_TOO_MANY_ITERATIONS)
		{
			std::cout << "Reached maximum number of function evaluations " << minimizer.get_failure_code() << std::endl;
			if(minimizer.obj_value_reduced())
			{
				std::cout << "Function value reduced" << std::endl;
				MDLHelper::updateShapeParameter(shapeIndex, numShapeVertices, MAX_PARAMETER_VARIATION, x, parameterVariation);
			}
			else
			{
				std::cout << "Function value not reduced" << std::endl;
			}
		}
		else
		{
			std::cout << "Minimizer failed convergence " << minimizer.get_failure_code() << std::endl;
		}

		MDLHelper::updateShapeData(shapeIndex, vecCs, matAs, matWs, sourcePointsVec, initialParam, parameterVariation, data);

		//Re-compute rigid alignment of iShape to mean of all other shapes
		if(!MDLHelper::alignShapeData(shapeIndex, excludedShapeMean, data, vecCs, matAs, matWs))
		{
			std::cout << "Failed to align shape after correspondence optimization" << std::endl;
			return;
		}
	}
}
bool MDLHelper::getRMMData(const std::string& sstrFileCollectionName, const std::string& sstrModelLmkIndexFileName, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName, const std::string & sstCorruptFileCollectionName
									, std::vector<double>& data, std::vector<std::string>& geometryFileNames, size_t& numIdentities, size_t& numExpressions, DataContainer& mesh, std::vector<size_t>& modelLmkIndices, std::vector<bool>& missingShapes
									, std::vector<bool>& corruptShapes, std::vector<std::vector<double>>& corruptData, std::vector<std::vector<std::pair<double,bool>>>& curruptDataLmks, std::vector<std::vector<double>>& corruptDataNormals, std::vector<std::vector<double>>& vecCs
									, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec, std::vector<size_t>& outerBoundaryVertexIDs, std::vector<size_t>& innerBoundaryVertexIDs)
{
	FileLoader loader;

	//Load the incomplete file collection
	if(!loader.loadIncompleteFileCollection(sstrFileCollectionName, data, mesh, geometryFileNames, missingShapes, numExpressions, numIdentities))
	{
		std::cout << "Unable to load the incomplete file collection " << sstrFileCollectionName << std::endl;
		return false;
	}

	//Load the model landmark indices
	const std::string sstrLmkIndexFileName = FileLoader::getFilePath(sstrModelLmkIndexFileName).empty() ? FileLoader::getFilePath(sstrFileCollectionName) + "/" + sstrModelLmkIndexFileName : sstrModelLmkIndexFileName;
	if(!loader.loadIndexFile(sstrLmkIndexFileName, modelLmkIndices))
	{
		std::cout << "Unable to load model landmark indices " << sstrLmkIndexFileName << std::endl;
		return false;
	}

	//Load the corrupt data and compute the corrupt data normals
	std::vector<std::string> dataFileNames;
	std::vector<DataContainer*> corruptDataMeshes;
	std::vector<std::vector<std::pair<double,bool>>> corruptDataLmks;
	if(!loader.loadCorruptFileCollection(sstCorruptFileCollectionName, dataFileNames, corruptDataMeshes, corruptDataLmks))
	{
		std::cout << "Unable to load corrupt data " << sstCorruptFileCollectionName << std::endl;
		return false;
	}

	const size_t numShapes = numIdentities*numExpressions;
	corruptShapes.resize(numShapes, false);
	corruptData.resize(numShapes);
	curruptDataLmks.resize(numShapes);
	corruptDataNormals.resize(numShapes);

	std::map<std::string, size_t> shapeIndexMap;
	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		shapeIndexMap.insert(std::make_pair(geometryFileNames[iShape], iShape));
	}

	const size_t numCorruptShapes = dataFileNames.size();
	for(size_t i = 0; i < numCorruptShapes; ++i)
	{
		const std::string sstrGeometryFile = dataFileNames[i]; 

		std::map<std::string, size_t>::const_iterator findIter = shapeIndexMap.find(sstrGeometryFile);
		if(findIter == shapeIndexMap.end())
		{
			std::cout << "Corresponding corrupt geometry file not found " << sstrGeometryFile << std::endl;
			return false;
		}

		const size_t shapeIndex = findIter->second;

		DataContainer* pMesh = corruptDataMeshes[i];
		const std::vector<double>& vertices = pMesh->getVertexList();
		corruptData[shapeIndex] = vertices;
		curruptDataLmks[shapeIndex] = corruptDataLmks[i];

		std::vector<double>& vertexNormals = corruptDataNormals[shapeIndex];
		MathHelper::computeVertexNormals(*pMesh, vertexNormals);

		corruptShapes[shapeIndex] = true;
		missingShapes[shapeIndex] = false;

		delete pMesh;
		corruptDataMeshes[i] = NULL;
	}


	//Load the thin-plate splines
	if(!loader.loadTpsFileCollection(sstrFileCollectionName, vecCs, matAs, matWs, sourcePointsVec))
	{
		std::cout << "Unable to load tps files for file collection " << sstrFileCollectionName << std::endl;
		return false;
	}

	//Load the vertices of the inner and outer mesh boundary if files are not empty
	if(!sstrOuterBoundaryIndexFileName.empty())
	{
		if(!loader.loadIndexFile(sstrOuterBoundaryIndexFileName, outerBoundaryVertexIDs))
		{
			std::cout << "Outer boundary vertex IDs not loaded " << sstrOuterBoundaryIndexFileName << std::endl;
			return false;
		}
	}
	else
	{
		std::cout << "WARNING: No outer boundary vertex IDs loaded " << sstrOuterBoundaryIndexFileName << std::endl;
	}

	if(!sstrInnerBoundaryIndexFileName.empty())
	{
		if(!loader.loadIndexFile(sstrInnerBoundaryIndexFileName, innerBoundaryVertexIDs))
		{
			std::cout << "Inner boundary vertex IDs not loaded " << sstrInnerBoundaryIndexFileName << std::endl;
			return false;
		}
	}
	else
	{
		std::cout << "WARNING: No inner boundary vertex IDs loaded " << sstrOuterBoundaryIndexFileName << std::endl;
	}

	return true;
}

void MDLHelper::convertParameters(const std::vector<std::vector<double>>& shapeParam, const size_t numShapes, const size_t shapeParameterDim, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, std::vector<double>& param)
{
	param.resize(numShapes*shapeParameterDim, 0.0);

	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		if(missingShapes[iShape] || corruptShapes[iShape])
			continue;

		const size_t shapeParamOffset = iShape*shapeParameterDim;

		const std::vector<double>& currSourcePoints = shapeParam[iShape];
		for(size_t i = 0; i < shapeParameterDim; ++i)
		{
			param[shapeParamOffset+i] = currSourcePoints[i];
		}
	}
}

bool MDLHelper::computeRMMOptimizationBounds(const size_t numShapes, const size_t shapeDataDim, const size_t shapeParameterDim, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes, const std::vector<double>& initialParam
														, const std::vector<size_t>& outerBoundaryVertexIDs, const std::vector<size_t>& innerBoundaryVertexIDs
														, vnl_vector<long>& completionBoundSelection, vnl_vector<long>& optimizationBoundSelection, std::vector<vnl_vector<double>>& lowerShapeBounds, std::vector<vnl_vector<double>>& upperShapeBounds)
{
	if(initialParam.size() != numShapes*shapeParameterDim)
	{
		return false;
	}

	completionBoundSelection = vnl_vector<long>(shapeDataDim, 2);
	optimizationBoundSelection = vnl_vector<long>(shapeParameterDim, 2);

	lowerShapeBounds.clear();
	lowerShapeBounds.resize(numShapes);

	upperShapeBounds.clear();
	upperShapeBounds.resize(numShapes);

	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		if(missingShapes[iShape] || corruptShapes[iShape])
		{
			lowerShapeBounds[iShape] = vnl_vector<double>(shapeDataDim, -MAX_COORD_VARIATON);
			upperShapeBounds[iShape] = vnl_vector<double>(shapeDataDim, MAX_COORD_VARIATON);
		}
		else
		{
			vnl_vector<double> lowerBounds(shapeParameterDim, 0.0);
			vnl_vector<double> upperBounds(shapeParameterDim, 0.0);

			const size_t shapeParamOffset = iShape*shapeParameterDim;
			for(size_t i = 0; i < shapeParameterDim; ++i)
			{
				const double currMin = OPTIMIZATION_DOMAIN_MIN-initialParam[shapeParamOffset+i];
				lowerBounds[i] = currMin;

				const double currMax = OPTIMIZATION_DOMAIN_MAX-initialParam[shapeParamOffset+i];
				upperBounds[i] = currMax;
			}

			const size_t numOuterBoundaryVertices = outerBoundaryVertexIDs.size();
			for(size_t i = 0; i < numOuterBoundaryVertices; ++i)
			{
				const size_t currVertexID = outerBoundaryVertexIDs[i];
				const size_t startIndex = 2*currVertexID;
				lowerBounds[startIndex] = -MAX_OUTER_BOUNDARY_VARIATION;
				lowerBounds[startIndex+1] = -MAX_OUTER_BOUNDARY_VARIATION;

				upperBounds[startIndex] = MAX_OUTER_BOUNDARY_VARIATION;
				upperBounds[startIndex+1] = MAX_OUTER_BOUNDARY_VARIATION;
			}

			const size_t numInnerBoundaryVertices = innerBoundaryVertexIDs.size();
			for(size_t i = 0; i < numInnerBoundaryVertices; ++i)
			{
				const size_t currVertexID = innerBoundaryVertexIDs[i];
				const size_t startIndex = 2*currVertexID;
				lowerBounds[startIndex] = -MAX_INNER_BOUNDARY_VARIATION;
				lowerBounds[startIndex+1] = -MAX_INNER_BOUNDARY_VARIATION;

				upperBounds[startIndex] = MAX_INNER_BOUNDARY_VARIATION;
				upperBounds[startIndex+1] = MAX_INNER_BOUNDARY_VARIATION;
			}

			lowerShapeBounds[iShape] = lowerBounds;
			upperShapeBounds[iShape] = upperBounds;
		}
	}

	return true;
}

void MDLHelper::computeKDTrees(const size_t numShapes, const std::vector<bool>& corruptShapes, const std::vector<std::vector<double>>& corruptData, std::vector<KDTree3*>& corrShapesKDTrees)
{
	corrShapesKDTrees.resize(numShapes, NULL);

	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		if(corruptShapes[iShape])
		{
			corrShapesKDTrees[iShape] = new KDTree3(corruptData[iShape]);
		}
	}
}

bool MDLHelper::computeCorruptDataLmkAlignment(const std::vector<double>& data, const size_t numShapes, const size_t shapeDataDim, const std::vector<bool>& corruptShapes, const std::vector<size_t>& modelLmkIndices
															, const std::vector<std::vector<std::pair<double,bool>>>& curruptDataLmks, std::vector<double>& corrShapes_s, std::vector<std::vector<double>>& corrShapes_R, std::vector<std::vector<double>>& corrShapes_t)
{
	corrShapes_s.resize(numShapes, 1);
	corrShapes_R.resize(numShapes);
	corrShapes_t.resize(numShapes);

	const size_t numLmks = modelLmkIndices.size();
	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		if(!corruptShapes[iShape])
		{
			continue;
		}

		const std::vector<std::pair<double,bool>>& dataLmks = curruptDataLmks[iShape];
		if(dataLmks.size() != 3*numLmks)
		{
			std::cout << "Wrong number of landmarks for shape " << iShape << " (" << dataLmks.size() << " != " << 3*numLmks << ")" << std::endl;
			return false;
		}

		const size_t shapeOffset = iShape*shapeDataDim;

		std::vector<double> tmpSourceLmks;
		std::vector<double> tmpTargetLmks;

		for(size_t iLmk = 0; iLmk < numLmks; ++iLmk)
		{
			const size_t lmkOffset = 3*iLmk;
			if(!dataLmks[lmkOffset].second || !dataLmks[lmkOffset+1].second || !dataLmks[lmkOffset].second)
			{
				continue;
			}

			const size_t sourceLmkIndex = modelLmkIndices[iLmk];
			tmpSourceLmks.push_back(data[shapeOffset+3*sourceLmkIndex]);
			tmpSourceLmks.push_back(data[shapeOffset+3*sourceLmkIndex+1]);
			tmpSourceLmks.push_back(data[shapeOffset+3*sourceLmkIndex+2]);

			tmpTargetLmks.push_back(dataLmks[lmkOffset].first);
			tmpTargetLmks.push_back(dataLmks[lmkOffset+1].first);
			tmpTargetLmks.push_back(dataLmks[lmkOffset+2].first);
		}

		double& s = corrShapes_s[iShape];
		std::vector<double>& R = corrShapes_R[iShape];
		std::vector<double>& t = corrShapes_t[iShape];
		if(!MathHelper::computeAlignmentTrafo(tmpSourceLmks, tmpTargetLmks, s, R, t, true))
		{
			std::cout << "Unable to compute initial transformation for shape " << iShape << std::endl;
			return false;
		}
	}

	return true;
}

void MDLHelper::initializeMissingData(std::vector<double>& data, const std::vector<size_t>& modeDim, const std::vector<bool>& missingShapes, const std::vector<bool>& corruptShapes)
{
	const size_t d1(modeDim[0]);
	const size_t d2(modeDim[1]);
	const size_t d3(modeDim[2]);
	const size_t numShapes = d2*d3;

	std::vector<std::vector<double>> modeSumShapes;
	modeSumShapes.resize(d2+d3);

	std::vector<size_t> modeSumShapesSizes;
	modeSumShapesSizes.resize(d2+d3, 0);

	//Compute sum of all valid expression scans for each identity
	for(size_t i2 = 0; i2 < d2; ++i2)
	{
		std::vector<double>& tmpMean = modeSumShapes[i2];
		tmpMean.resize(d1, 0.0);

		size_t& numValidShapes = modeSumShapesSizes[i2];
		
		for(size_t i3 = 0; i3 < d3; ++i3)
		{
			const size_t iShape = i3*d2+i2;
			if(missingShapes[iShape] || corruptShapes[iShape])
			{
				//Current shape is missing or corrupt, hence continue
				continue;
			}

			const size_t shapeOffset = iShape*d1;
			for(size_t i1 = 0; i1 < d1; ++i1)
			{
				tmpMean[i1] += data[shapeOffset+i1];
			}

			++numValidShapes;
		}
	}

	//Compute sum of all valid identity scans for each expression
	for(size_t i3 = 0; i3 < d3; ++i3)
	{
		std::vector<double>& tmpMean = modeSumShapes[d2+i3];
		tmpMean.resize(d1, 0.0);

		size_t& numValidShapes = modeSumShapesSizes[d2+i3];
		
		for(size_t i2 = 0; i2 < d2; ++i2)
		{
			const size_t iShape = i3*d2+i2;
			if(missingShapes[iShape] || corruptShapes[iShape])
			{
				//Current shape is missing or corrupt, hence continue
				continue;
			}

			const size_t shapeOffset = iShape*d1;
			for(size_t i1 = 0; i1 < d1; ++i1)
			{
				tmpMean[i1] += data[shapeOffset+i1];
			}

			++numValidShapes;
		}
	}

	//Fill missing values by mean of all scans that are from the same subject or the same expression
	for(size_t iShape = 0; iShape < numShapes; ++iShape)
	{
		if(!missingShapes[iShape] && !corruptShapes[iShape])
		{
			//Current shape is missing or corrupt, hence continue
			continue;
		}

		const size_t i2 = iShape%d2;
		const size_t i3 = (iShape-i2)/d2;

		if(modeSumShapesSizes[i2] == 0 && modeSumShapesSizes[d2+i3] == 0)
		{
			//No other scans of the subject or of the expression available
			continue;
		}

		const size_t shapeOffset = iShape*d1;

		if(modeSumShapesSizes[i2] == 0)
		{
			const double factor = 1.0/static_cast<double>(modeSumShapesSizes[d2+i3]);

			const std::vector<double>& sum = modeSumShapes[d2+i3];

			for(size_t i1 = 0; i1 < d1; ++i1)		
			{
				const double currValue = factor*sum[i1];
				data[shapeOffset+i1] = currValue;
			}
		}
		else if(modeSumShapesSizes[d2+i3] == 0)
		{
			const double factor = 1.0/static_cast<double>(modeSumShapesSizes[i2]);

			const std::vector<double>& sum = modeSumShapes[i2];

			for(size_t i1 = 0; i1 < d1; ++i1)		
			{
				const double currValue = factor*sum[i1];
				data[shapeOffset+i1] = currValue;
			}
		}
		else
		{
			const double factor1 = 1.0/static_cast<double>(modeSumShapesSizes[i2]);
			const double factor2 = 1.0/static_cast<double>(modeSumShapesSizes[d2+i3]);

			const std::vector<double>& sum1 = modeSumShapes[i2];
			const std::vector<double>& sum2 = modeSumShapes[d2+i3];

			for(size_t i1 = 0; i1 < d1; ++i1)		
			{
				const double currValue = 0.5*(factor1*sum1[i1]+factor2*sum2[i1]);
				data[shapeOffset+i1] = currValue;
			}
		}
	}
}

void MDLHelper::getBoundaryNeighbors(const DataContainer& mesh, const std::vector<size_t>& boundaryVertexIDs, std::vector<size_t>& neighborsVertexIDs)
{
	const size_t numVertices = mesh.getNumVertices();

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	const size_t numTriangles = vertexIndexList.size();

	std::vector<std::set<int>> vertexNeighbors;
	vertexNeighbors.resize(numVertices);

	for(size_t i = 0; i < numTriangles; ++i)
	{
		const std::vector<int>& currTriangleIndices = vertexIndexList[i];
		const int i1 = currTriangleIndices[0];
		const int i2 = currTriangleIndices[1];
		const int i3 = currTriangleIndices[2];

		vertexNeighbors[i1].insert(i2);
		vertexNeighbors[i1].insert(i3);

		vertexNeighbors[i2].insert(i1);
		vertexNeighbors[i2].insert(i3);

		vertexNeighbors[i3].insert(i1);
		vertexNeighbors[i3].insert(i2);
	}

	std::set<size_t> tmpNeighbors;

	const size_t numBoundaryVertices = boundaryVertexIDs.size();
	for(size_t i = 0; i < numBoundaryVertices; ++i)
	{
		const size_t currIndex = boundaryVertexIDs[i];

		const std::set<int>& currNeighbors = vertexNeighbors[currIndex];
		if(currNeighbors.empty())
		{
			continue;
		}

		//Iterate over all neighbors
		std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
		std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			const size_t currNeighborIndex = static_cast<size_t>(*currNeighborIter);
			tmpNeighbors.insert(currNeighborIndex);
		}
	}

	neighborsVertexIDs.reserve(tmpNeighbors.size());

	std::set<size_t>::const_iterator tmpNeighborIter = tmpNeighbors.begin();
	std::set<size_t>::const_iterator endTmpNeighborIter = tmpNeighbors.end();
	for(; tmpNeighborIter != endTmpNeighborIter; ++tmpNeighborIter)
	{
		neighborsVertexIDs.push_back(*tmpNeighborIter);
	}
}

void MDLHelper::outputConfig(const std::string& sstrOutFolder)
{
	const std::string sstrFileName = sstrOutFolder + "/Config.txt";
	std::fstream outStream(sstrFileName, std::ios::out);

	outStream << std::endl;
	outStream << "Number of Iterations: " << NUM_ITERATION << std::endl;
	outStream << "Number of Function Evaluation: " << NUM_FKT_EVAL << std::endl;
	outStream << "Recompute Alignment after " << NUM_NUM_FKT_EVAL_ALIGNMENT << " Function Evaluation" << std::endl;
	outStream << std::endl;
	outStream << "Identity weight: " << IDENTITY_WEIGHT << std::endl;
	outStream << "Expression weight: " << EXPRESSION_WEIGHT << std::endl;
	outStream << "Corrupt data weight: " << CORRUPT_DATA_WEIGHT << std::endl;
	outStream << "Regularization weight (data completion): " << DATA_COMPLETION_SMOOTHNESS_WEIGHT << std::endl;
	outStream << "Regularization weight (correspondence optimization): " << CORRESPONDENCE_OPTIMIZATION_SMOOTHNESS_WEIGHT << std::endl;

#ifdef USE_BI_LAPLACIAN_SMOOTHNESS
	outStream << "Use of Bi-Laplacian Smoothing" << std::endl;
#else
	outStream << "Use of Laplacian Smoothing" << std::endl;
#endif
	
	outStream << std::endl;
	outStream << "Corrupt data distance threshold: " << MAX_CORR_DATA_DIST << std::endl;
	outStream << "Max vertex variation during optimization: " << MAX_COORD_VARIATON << std::endl;
	outStream << "Max parameter variation during optimization: " << MAX_PARAMETER_VARIATION << std::endl;
	outStream << "Maximum variation outer boundary: " << MAX_OUTER_BOUNDARY_VARIATION << std::endl;
	outStream << "Maximum variation inner boundary: " << MAX_INNER_BOUNDARY_VARIATION << std::endl;

	outStream << std::endl;
	outStream.close();
}

void MDLHelper::outputCompactness(const std::vector<double>& data, const std::vector<size_t>& semCorr, const size_t d1, const size_t d2, const size_t d3, const std::string& sstrOutFileName)
{
	std::fstream compactnessStream(sstrOutFileName, std::ios::out);

#ifdef DEBUG_OUTPUT
	std::cout << "Output compactness" << std::endl;
#endif

	//Center data
	std::vector<double> centeredData;
	std::vector<double> mean;
	MathHelper::centerData(data, d1, centeredData, mean);

	std::vector<size_t> modeDim;
	modeDim.push_back(d1);
	modeDim.push_back(d2);
	modeDim.push_back(d3);

	double sumMode2(0.0);
	double sumSVMode2(0.0);
	double avgLogSumMode2(0.0);

	double sumMode3(0.0);
	double sumSVMode3(0.0);
	double avgLogSumMode3(0.0);

	//Mode 2
	{
		compactnessStream << "Mode 2" << std::endl;

#ifdef DEBUG_OUTPUT
		std::cout << "Mode 2" << std::endl;
#endif

		size_t mode2Dim(0);
		std::vector<double> mode2Cov;
		if(!MathHelper::computeCovarianceMatrix(centeredData, semCorr, modeDim, 2, mode2Cov, mode2Dim))
		{
			std::cout << "Failed computing covariance matrix" << std::endl;
			return;
		}

		std::vector<double> mode2EVs;
		MathHelper::computeSingularValues(mode2Cov, mode2Dim, mode2Dim, mode2EVs);

#ifdef DEBUG_OUTPUT
		std::cout << "Num eigenvalues: " << mode2EVs.size() << std::endl;
		std::cout << "EV: ";
#endif
		for(size_t i = 0; i < mode2EVs.size(); ++i)
		{
			sumMode2 += mode2EVs[i];
			sumSVMode2 += sqrt(d3*mode2EVs[i]);
			avgLogSumMode2 += log(mode2EVs[i]);

#ifdef DEBUG_OUTPUT
			std::cout << std::setprecision(3) << mode2EVs[i] << " ";
#endif	
		}

		avgLogSumMode2 /= static_cast<double>(mode2EVs.size());

#ifdef DEBUG_OUTPUT
		std::cout << std::endl;
#endif

		double tmpSumMode2(0.0);
		for(size_t i = 0; i < mode2EVs.size(); ++i)
		{
			tmpSumMode2 += mode2EVs[i];

			const double currValue = 100.0*tmpSumMode2/sumMode2;
			compactnessStream <<  tmpSumMode2 << " (" << currValue << ") " << std::endl;

#ifdef DEBUG_OUTPUT
			std::cout << std::setprecision(3) << tmpSumMode2 << " (" << currValue << ") " << std::endl;
#endif
		}

	}

	std::cout << std::endl;

	//Mode 3
	{
		compactnessStream << "Mode 3" << std::endl;

#ifdef DEBUG_OUTPUT
		std::cout << "Mode 3" << std::endl;
#endif

		size_t mode3Dim(0);
		std::vector<double> mode3Cov;
		if(!MathHelper::computeCovarianceMatrix(centeredData, semCorr, modeDim, 3, mode3Cov, mode3Dim))
		{
			std::cout << "Failed computing covariance matrix" << std::endl;
			return;
		}

		std::vector<double> mode3EVs;
		MathHelper::computeSingularValues(mode3Cov, mode3Dim, mode3Dim, mode3EVs);

#ifdef DEBUG_OUTPUT
		std::cout << "Num eigenvalues: " << mode3EVs.size() << std::endl;
		std::cout << "EV: ";
#endif

		for(size_t i = 0; i < mode3EVs.size(); ++i)
		{
			sumMode3 += mode3EVs[i];
			sumSVMode3 += sqrt(d2*mode3EVs[i]);
			avgLogSumMode3 += log(mode3EVs[i]);

#ifdef DEBUG_OUTPUT
			std::cout << std::setprecision(3) << mode3EVs[i] << " ";
#endif	
		}

		avgLogSumMode3 /= static_cast<double>(mode3EVs.size());

#ifdef DEBUG_OUTPUT
		std::cout << std::endl;
#endif

		double tmpSumMode3(0.0);
		for(size_t i = 0; i < mode3EVs.size(); ++i)
		{
			tmpSumMode3 += mode3EVs[i];

			const double currValue = 100.0*tmpSumMode3/sumMode3;
			compactnessStream <<  tmpSumMode3 << " (" << currValue << ") " << std::endl;

#ifdef DEBUG_OUTPUT
			std::cout << std::setprecision(3) << tmpSumMode3 << " (" << currValue << ") " << std::endl;
#endif
		}
	}

#ifdef DEBUG_OUTPUT
	std::cout << std::endl;
#endif

	compactnessStream << std::endl;
	compactnessStream << "Compactness (sum): " << sumMode2+sumMode3 << std::endl;
	compactnessStream << std::endl;
	compactnessStream << "Nuclear tensor norm: " << sumSVMode2+sumSVMode3 << " (" << sumSVMode2 << " + " << sumSVMode3 << ")" << std::endl;
	compactnessStream << std::endl;
	
	compactnessStream << "Average log sum identity: " << avgLogSumMode2 << std::endl;
	compactnessStream << "Average log sum expression: " << avgLogSumMode3 << std::endl;
	compactnessStream << "Average log sum: " << avgLogSumMode2+avgLogSumMode3 << std::endl;

	compactnessStream.close();
}

std::string MDLHelper::getConfigOutFolder(const std::string& sstrOutFolder)
{
	std::stringstream outIdentityWeight;
	outIdentityWeight << IDENTITY_WEIGHT;

	std::stringstream outExpressionWeight;
	outExpressionWeight << EXPRESSION_WEIGHT;

	std::stringstream outSmoothnessWeight;
	outSmoothnessWeight << CORRESPONDENCE_OPTIMIZATION_SMOOTHNESS_WEIGHT;

#ifdef USE_BI_LAPLACIAN_SMOOTHNESS
	std::string sstrSmoothnessPrefix = "_bls";
#else
	std::string sstrSmoothnessPrefix = "_ls";
#endif

	return sstrOutFolder + "_id" + outIdentityWeight.str() + "_exp" + outExpressionWeight.str() + sstrSmoothnessPrefix + outSmoothnessWeight.str();
}