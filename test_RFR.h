//
//  test_RFR.h
//  RandomFeatureRegistration
//
//  Created by Lili on 11/27/16.
//  Copyright (c) 2016 UBC. All rights reserved.
//

#ifndef __RandomFeatureRegistration__test_RFR__
#define __RandomFeatureRegistration__test_RFR__

#include <iostream>
#include "RFRTreeNode.h"
#include "RFR_util.h"
#include "RFRTree.h"
#include "cvxIO.hpp"
#include "KLT_Util.h"
#include "ms7ScenesUtil.hpp"
#include "cvxPoseEstimation.hpp"
#include "Kabsch.h"
#include "fstream"



void test_RandomFeatureMatching();

void test_KLT_tracking();

void test_FAST_KLT_VO();

void test_GoodFeatures_KLT_VO();

void test_RandomFeature_KLT();


#endif /* defined(__RandomFeatureRegistration__test_RFR__) */


