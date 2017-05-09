#ifndef KFOLDVALIDATION_HPP
#define KFOLDVALIDATION_HPP

#include <opencv2/core/core.hpp>

class KFoldValidation
{
public:
	KFoldValidation();
	~KFoldValidation();

	int test();
};
#endif