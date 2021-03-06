/*************************************************************************************************************************/
// This source is provided for NON-COMMERCIAL RESEARCH PURPOSES only, and is provided �as is� WITHOUT ANY WARRANTY; 
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

#ifndef PERFORMANCECOUNTER_H
#define PERFORMANCECOUNTER_H

class PerformanceCounter
{
public:
	static double getTime();

private:
	PerformanceCounter();

	~PerformanceCounter();

	PerformanceCounter(const PerformanceCounter& costFunction);

	PerformanceCounter& operator=(const PerformanceCounter& costFunction);
};

#endif