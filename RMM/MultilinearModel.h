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

#ifndef MULTILINEARMODEL_H
#define MULTILINEARMODEL_H

#include <string>
#include <vector>
#include <assert.h>
#include <stdlib.h>

template<typename T>
class MultiArray4
{
public:
	MultiArray4(size_t d1, size_t d2 = 1, size_t d3 = 1, size_t d4 = 1)
	: m_d1(d1)
	, m_d2(d2)
	, m_d3(d3)
	, m_d4(d4)
	{
		m_data.resize(d1*d2*d3*d4);
	}

	~MultiArray4()
	{	}

	void setElements(const std::vector<T>& data)
	{
		size_t index(0);
		for(int i = 0; i < m_d4; ++i)
		{
			for(int j = 0; j < m_d3; ++j)
			{
				for(int k = 0; k < m_d2; ++k)
				{
					for(int l = 0; l < m_d1; ++l)
					{
						setElementAt(data[index], l, k, j, i);
						++index;
					}
				}
			}
		}
	}

	T getElementAt(size_t i1, size_t i2 = 0, size_t i3 = 0, size_t i4 = 0) const
	{
		const size_t index = getIndex(i1, i2, i3, i4);
		return m_data[index];
	}

	void getElements(std::vector<T>& data) const
	{
		data.clear();
		data.reserve(m_d1*m_d2*m_d3*m_d4);

		for(int i = 0; i < m_d4; ++i)
		{
			for(int j = 0; j < m_d3; ++j)
			{
				for(int k = 0; k < m_d2; ++k)
				{
					for(int l = 0; l < m_d1; ++l)
					{
						const double tensorElement =  getElementAt(l, k, j, i);
						data.push_back(tensorElement);
					}
				}
			}
		}
	}

	void setElementAt(const T value, size_t i1, size_t i2 = 0, size_t i3 = 0, size_t i4 = 0)
	{
		const size_t index = getIndex(i1, i2, i3, i4);
		m_data[index] = value;
	}

	size_t getSize() const
	{
		return m_data.size();
	}

private:
	size_t getIndex(const size_t i1, const size_t i2, const size_t i3, const size_t i4) const
	{
		assert(i1 < m_d1);
		assert(i2 < m_d2);
		assert(i3 < m_d3);
		assert(i4 < m_d4);
		return i1+i2*m_d1+i3*m_d1*m_d2+i4*m_d1*m_d2*m_d3;
	}

	size_t m_d1;
	size_t m_d2;
	size_t m_d3;
	size_t m_d4;

	std::vector<T> m_data;
};


class Tensor
{
	typedef MultiArray4<double> multiArray;

public:
	Tensor();

	~Tensor();

	void clear();

	void init(const std::vector<double>& data, size_t d1, size_t d2, size_t d3, size_t d4 = 1);

	void init(size_t d1, size_t d2, size_t d3, size_t d4 = 1);

	double getElement(size_t i1, size_t i2, size_t i3, size_t i4 = 0) const;

	void getElements(std::vector<double>& data) const;

	void setElement(const double value, size_t i1, size_t i2, size_t i3, size_t i4 = 0);

	void modeMultiply(const std::vector<double>& matrixU, const std::string& type, const size_t numRows, const size_t numColumns, const size_t mode, Tensor& outTensor) const;

	void unfold(const size_t mode, std::vector<double>& unfoldedTensor) const;

	size_t getModeDimension(const size_t mode) const
	{
		const size_t val = mode>0 && mode-1 < m_modeDims.size() ? m_modeDims[mode-1] : 0;
		return val;
	}

	std::vector<size_t> getModeDimensions() const
	{
		return m_modeDims;
	}

	bool isEmpty() const
	{
		if(m_pTensor == NULL)
		{
			return true;
		}

		return m_pTensor->getSize() == 0;
	}

private:

	void setModeDimensions(size_t d1, size_t d2, size_t d3, size_t d4)
	{
		m_modeDims.clear();
		m_modeDims.push_back(d1);
		m_modeDims.push_back(d2);
		m_modeDims.push_back(d3);
		m_modeDims.push_back(d4);
	}

	std::vector<size_t> m_modeDims;

	multiArray* m_pTensor;
};

#endif