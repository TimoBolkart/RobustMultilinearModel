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

#include "MultilinearModel.h"
#include <string>

Tensor::Tensor()
: m_modeDims()
, m_pTensor(NULL)
{

}

Tensor::~Tensor()
{
	clear();
}

void Tensor::clear()
{
	setModeDimensions(0, 0, 0, 0);

	if(m_pTensor!=NULL)
	{
		delete m_pTensor;
		m_pTensor = NULL;
	}
}

void Tensor::init(const std::vector<double>& data, size_t d1, size_t d2, size_t d3, size_t d4)
{
	init(d1, d2, d3, d4);
	m_pTensor->setElements(data);
}

void Tensor::init(size_t d1, size_t d2, size_t d3, size_t d4)
{
	clear();

	setModeDimensions(d1, d2, d3, d4);

	assert(m_pTensor == NULL);
	m_pTensor = new multiArray(d1, d2, d3, d4);
}

double Tensor::getElement(size_t i1, size_t i2, size_t i3, size_t i4) const
{	
	return m_pTensor->getElementAt(i1, i2, i3,i4);
}

void Tensor::getElements(std::vector<double>& data) const
{
	return m_pTensor->getElements(data);
}

void Tensor::setElement(const double value, size_t i1, size_t i2, size_t i3, size_t i4)
{
	m_pTensor->setElementAt(value, i1, i2, i3, i4);
}

void Tensor::modeMultiply(const std::vector<double>& matrixU, const std::string& type, const size_t numRows, const size_t numColumns, const size_t mode, Tensor& outTensor) const
{	
	if(type!="N" && type!="T")
	{
		return;
	}

	const size_t e1 = (mode==1) ? (type=="N" ? numRows : numColumns) : getModeDimension(1); 
	const size_t e2 = (mode==2) ? (type=="N" ? numRows : numColumns) : getModeDimension(2);
	const size_t e3 = (mode==3) ? (type=="N" ? numRows : numColumns) : getModeDimension(3);
	const size_t e4 = (mode==4) ? (type=="N" ? numRows : numColumns) : getModeDimension(4);

	outTensor.init(e1, e2, e3, e4);

	for(size_t i4 = 0; i4 < e4; ++i4)
	{
		for(size_t i3 = 0; i3 < e3; ++i3)
		{
			for(size_t i2 = 0; i2 < e2; ++i2)
			{
				for(size_t i1 = 0; i1 < e1; ++i1)
				{
					double tmpValue(0.0);
					if(mode==1)
					{
						for(int l = 0; l < getModeDimension(1); ++l)
						{
							if(type=="N")
							{
								tmpValue += m_pTensor->getElementAt(l, i2, i3, i4)*matrixU[i1+l*numRows]; //matrixU[i1][l];  
							}
							else if(type=="T")
							{
								tmpValue += m_pTensor->getElementAt(l, i2, i3, i4)*matrixU[i1*numRows+l]; //matrixU^T[l][i1];
							}
						}
					}
					else if(mode==2)
					{
						for(int l = 0; l < getModeDimension(2); ++l)
						{
							if(type=="N")
							{
								tmpValue += m_pTensor->getElementAt(i1, l, i3, i4)*matrixU[i2+l*numRows]; //matrixU[i2][l];
							}
							else if(type=="T")
							{
								tmpValue += m_pTensor->getElementAt(i1, l, i3, i4)*matrixU[i2*numRows+l]; //matrixU^T[l][i2];
							}
						}
					}
					else if(mode==3)
					{
						for(int l = 0; l < getModeDimension(3); ++l)
						{
							if(type=="N")
							{
								tmpValue += m_pTensor->getElementAt(i1, i2, l, i4)*matrixU[i3+l*numRows]; //matrixU[i3][l];
							}
							else if(type=="T")
							{
								tmpValue += m_pTensor->getElementAt(i1, i2, l, i4)*matrixU[i3*numRows+l]; //matrixU^T[l][i3];
							}
						}
					}
					else if(mode==4)
					{
						for(int l = 0; l < getModeDimension(4); ++l)
						{
							if(type=="N")
							{
								tmpValue += m_pTensor->getElementAt(i1, i2, i3, l)*matrixU[i4+l*numRows]; //matrixU[i4][l];
							}
							else if(type=="T")
							{
								tmpValue += m_pTensor->getElementAt(i1, i2, i3, l)*matrixU[i4*numRows+l]; //matrixU^T[l][i4];
							}
						}
					}

					outTensor.setElement(tmpValue, i1, i2, i3, i4);
				}
			}
		}
	}
}

void Tensor::unfold(const size_t mode, std::vector<double>& unfoldedTensor) const
{
	//if(mode==1)
	//{
	//	e1 =  d1; e2 =  d2; e3 =  d3; e4 =  d4;
	//}
	//else if(mode==2)
	//{
	//	e1 =  d2; e2 =  d3; e3 =  d4; e4 =  d1;
	//}
	//else if(mode==3)
	//{
	//	e1 =  d3; e2 =  d4; e3 =  d1; e4 =  d2;
	//}
	//else if(mode==4)
	//{
	//	e1 =  d4; e2 =  d1; e3 =  d2; e4 =  d3;
	//}

	const size_t e1 = getModeDimension(mode);
	const size_t e2 = getModeDimension((mode+1)%5+(mode+1)/5);
	const size_t e3 = getModeDimension((mode+2)%5+(mode+2)/5);
	const size_t e4 = getModeDimension((mode+3)%5+(mode+3)/5);

	unfoldedTensor.clear();
	unfoldedTensor.reserve(e1*e2*e3*e4);

	for(int i4 = 0; i4 < e4; ++i4)
	{
		for(int i3 = 0; i3 < e3; ++i3)
		{
			for(int i2 = 0; i2 < e2; ++i2)
			{
				for(int i1 = 0; i1 < e1; ++i1)
				{
					if(mode==1)
					{
						unfoldedTensor.push_back(m_pTensor->getElementAt(i1, i2, i3, i4));
					}
					else if(mode==2)
					{
						unfoldedTensor.push_back(m_pTensor->getElementAt(i4, i1, i2, i3));
					}
					else if(mode==3)
					{
						unfoldedTensor.push_back(m_pTensor->getElementAt(i3, i4, i1, i2));
					}
					else if(mode==4)
					{
						unfoldedTensor.push_back(m_pTensor->getElementAt(i2, i3, i4, i1));
					}
				}
			}
		}
	}
}