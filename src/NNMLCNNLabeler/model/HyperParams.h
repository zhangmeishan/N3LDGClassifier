#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams {

    int batch;

    dtype nnRegular; // for optimization
    dtype adaAlpha;  // for optimization
    dtype adaEps; // for optimization

    int hiddenSize;
    int cnnLayerSize;
    int wordContext;
    int wordWindow;
    int windowOutput_layer1;
    int windowOutput_layer2;
    dtype dropProb;


    //auto generated
    int wordDim;
    int inputSize;
    int labelSize;

  public:
    HyperParams() {
        bAssigned = false;
        batch = 1;
        cnnLayerSize = 1;
    }

  public:
    void setRequared(Options& opt) {
        nnRegular = opt.regParameter;
        adaAlpha = opt.adaAlpha;
        adaEps = opt.adaEps;
        hiddenSize = opt.hiddenSize;
        wordContext = opt.wordcontext;
        dropProb = opt.dropProb;
        batch = opt.batchSize;

        if (opt.cnnLayerSize < 1)
            cnnLayerSize = 1;
        else
            cnnLayerSize = opt.cnnLayerSize;

        bAssigned = true;
    }

    void clear() {
        bAssigned = false;
    }

    bool bValid() {
        return bAssigned;
    }


    void saveModel(std::ofstream &os) const {
        os << nnRegular << std::endl;
        os << adaAlpha << std::endl;
        os << adaEps << std::endl;

        os << hiddenSize << std::endl;
        os << wordContext << std::endl;
        os << wordWindow << std::endl;
        os << windowOutput_layer1 << std::endl;
        os << dropProb << std::endl;


        os << wordDim << std::endl;
        os << inputSize << std::endl;
        os << labelSize << std::endl;
    }

    void loadModel(std::ifstream &is) {
        is >> nnRegular;
        is >> adaAlpha;
        is >> adaEps;

        is >> hiddenSize;
        is >> wordContext;
        is >> wordWindow;
        is >> windowOutput_layer1;
        is >> dropProb;


        is >> wordDim;
        is >> inputSize;
        is >> labelSize;

        bAssigned = true;
    }
  public:

    void print() {

    }

  private:
    bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */