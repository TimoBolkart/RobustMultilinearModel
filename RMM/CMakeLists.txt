SET( RMM_Files
	ExpressionPermutationSA.cpp
	FileLoader.cpp
	FileWriter.cpp
	KDTree3.cpp
	MathHelper.cpp
	MDLHelper.cpp
	MDLShapeCostFunction.cpp
	MultilinearModel.cpp
	PerformanceCounter.cpp
	RelabelMultilinearData.cpp
	TensorCompletionCostFunction.cpp
	Main.cpp
)

INCLUDE_DIRECTORIES(${ANN_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CLAPACK_INCLUDE_DIR}) 
INCLUDE_DIRECTORIES(${ITK_INCLUDES}) 
   
ADD_EXECUTABLE(RMM ${RMM_Files})
TARGET_LINK_LIBRARIES(RMM ${ANN_LIBRARIES} ${ITK_LIBRARIES} ${CLAPACK_LIBRARIES})   